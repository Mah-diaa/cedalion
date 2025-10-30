"""Solver for the image reconstruction problem."""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pint
import xarray as xr
from scipy.spatial import KDTree
from tqdm import tqdm

import cedalion.dot.forward_model as fwm
import cedalion.io.utils as ioutils
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import nirs, units
from cedalion.dot.head_model import TwoSurfaceHeadModel

logger = logging.getLogger("cedalion")

@dataclass
class RegularizationParams:
    """Parameters controlling the regularization of the inverse problem.

    Args:
        alpha_meas: ...
        ...
    """

    alpha_meas: float
    alpha_spatial: None | float
    apply_c_meas: bool  # FIXME better name: measurement regularization


# FIXME define the proper interface
class SpatialBasisFunctions(ABC):
    """Parameters controlling the spatial basis functions.

    Args:
        threshold_brain: ...
        ...
    """

    @abstractmethod
    def to_file(self, fname : Path | str):
        """Serialize prepared spatial basis functions into a file.

        Args:
            fname: path of the output file.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, fname : Path | str) -> "SpatialBasisFunctions":
        """Load prepared spatial basis functions from a file.

        Args:
            fname: path of the file to read from.

        Returns:
            SpatialBasisFunctions: Loaded spatial basis functions instance.
        """
        pass


class GaussianSpatialBasisFunctions(SpatialBasisFunctions):
    def __init__(
        self,
        head_model: TwoSurfaceHeadModel,
        Adot: xr.DataArray,
        threshold_brain: cdt.QLength,
        threshold_scalp: cdt.QLength,
        sigma_brain: cdt.QLength,
        sigma_scalp: cdt.QLength,
        mask_threshold: float,
    ):
        """Gaussian Spatial Basis Functions.

        TBD

        Args:
            head_model: TBD
            Adot : TBD
            threshold_brain: TBD
            threshold_scalp: TBD
            sigma_brain: TBD
            sigma_scalp : TBD
            mask_threshold: TBD, mention log scale
        """
        self.threshold_brain = threshold_brain
        self.threshold_scalp = threshold_scalp
        self.sigma_brain = sigma_brain
        self.sigma_scalp = sigma_scalp
        self.mask_threshold = mask_threshold

        self._mask : xr.DataArray = None # shape (vertex)
        self.G_brain : xr.DataArray = None # shape (vertex, kernel)
        self.G_scalp : xr.DataArray = None  # shape (vertex, kernel)
        self.H: xr.DataArray = None  # shape(channel, kernel, wavelength)

        # compute _G
        self._compute_sensitivity_mask(Adot)
        self._compute_G_gaussian_kernels(head_model)
        self._compute_H(Adot)


    def _compute_sensitivity_mask(self, Adot, wavelength_idx: int = 0):
        """Compute sensitivity mask based on intensity threshold.

        The mask selects those vertices whose summed contribution to the sensitivity
        of all channels is above 10^(mask_threshold).

        Args:
            Adot: Sensitivity matrix.
            wavelength_idx: Index of wavelength to use for mask computation.
        """

        intensity = np.log10(
            Adot.isel(wavelength=wavelength_idx)
            .sum("channel")
            .clip(min=10 ** (self.mask_threshold - 1))  # avoid log10(0)
        )

        mask = intensity > self.mask_threshold
        mask = mask.drop_vars("wavelength")  # keep the is_brain coordinate
        self._mask = mask

    def _downsample_mesh(
        self, mesh: xr.DataArray, threshold: cdt.QLength, mask: np.ndarray
    ) -> xr.DataArray:
        """Downsample the mesh to get seeds of spatial bases.

        Args:
            mesh: vertices of either the brain or scalp surface.
            threshold: distance between vertices in downsampled mesh.
            mask: boolean mask to select mesh vertices

        Returns:
            downsampled mesh vertices as a xr.DataArray

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Downsample the mesh using the specified method
        mesh_units = mesh.pint.units
        threshold = threshold.to(mesh_units).magnitude

        mesh = mesh.rename({"label": "vertex"}).pint.dequantify()
        mesh_masked = mesh[mask, :]
        mesh_new = []

        for vv in tqdm(mesh_masked):
            if len(mesh_new) == 0:
                mesh_new.append(vv)
                tree = KDTree(mesh_new)  # Build KDTree for the first point
                continue

            # Query the nearest neighbor within the threshold
            distance, _ = tree.query(vv, distance_upper_bound=threshold)

            # If no point is within the threshold, append the new point
            if distance == float("inf"):
                mesh_new.append(vv)
                tree = KDTree(mesh_new)  # Rebuild the KDTree with the new point

        mesh_new_xr = xr.DataArray(
            mesh_new,
            dims=mesh.dims,
            coords={"vertex": np.arange(len(mesh_new))},
            attrs={"units": mesh_units},
        )

        mesh_new_xr = mesh_new_xr.pint.quantify()

        return mesh_new_xr


    def _get_gaussian_kernels_on_mesh(
        self, mesh_downsampled: xr.DataArray, mesh: xr.DataArray, sigma: cdt.QLength
    ):
        """Compute the matrix containing the spatial bases.

        Args:
            mesh_downsampled: vertices of either the downsampled brain or scalp surface.
                This is used to define the centers of the spatial bases.
            mesh: the original fully sampeld mesh vertices of the brain or scalp.
            sigma: standard deviation used for defining the Gaussian kernel.

        Returns:
            xr.DataArray: matrix containing the spatial bases

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Create Gaussian kernels based on the mesh and parameters
        assert mesh.pint.units == mesh_downsampled.pint.units

        mesh_units = mesh.pint.units
        sigma = sigma.to(mesh_units).magnitude

        # Covariance matrix
        cov_matrix = (sigma**2) * np.eye(3)
        inv_cov = np.linalg.inv(cov_matrix)  # Inverse of Cov_matrix
        det_cov = np.linalg.det(cov_matrix)  # Determinant of Cov_matrix
        denominator = np.sqrt((2 * np.pi) ** 3 * det_cov)  # Pre-calculate denominator

        mesh_downsampled = mesh_downsampled.pint.dequantify().values
        mesh = mesh.pint.dequantify().values

        diffs = mesh_downsampled[:, None, :] - mesh[None, :, :]

        # Efficient matrix multiplication using np.einsum to compute
        # (x-mu)' * inv_cov * (x-mu) for all pairs
        exponents = -0.5 * np.einsum("ijk,kl,ijl->ij", diffs, inv_cov, diffs)

        # Compute the kernel matrix
        kernel_matrix = np.exp(exponents) / denominator
        n_vertex = mesh.shape[0]

        dimensions = kernel_matrix.shape

        if dimensions[0] != n_vertex:
            dims = ["kernel", "vertex"]
            #n_kernel = dimensions[0]
        else:
            dims = ["vertex", "kernel"]
            #n_kernel = dimensions[1]

        kernel_matrix_xr = xr.DataArray(
            kernel_matrix,
            dims=dims,
            #coords={"vertex": np.arange(n_vertex), "kernel": np.arange(n_kernel)},
        )

        kernel_matrix_xr = kernel_matrix_xr.transpose("vertex", "kernel")
        return kernel_matrix_xr


    def _compute_G_gaussian_kernels(self, head_model : TwoSurfaceHeadModel):
        """Compute the G matrix which contains all the information of the spatial basis.

        Args:
            head_model: Head model with brain and scalp surfaces.

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024
        """
        brain_downsampled = self._downsample_mesh(
            head_model.brain.vertices,
            self.threshold_brain,
            self._mask.sel(vertex=self._mask.is_brain),
        )
        scalp_downsampled = self._downsample_mesh(
            head_model.scalp.vertices,
            self.threshold_scalp,
            self._mask.sel(vertex=~self._mask.is_brain),
        )

        self.G_brain = self._get_gaussian_kernels_on_mesh(
            brain_downsampled, head_model.brain.vertices, self.sigma_brain
        )
        self.G_scalp = self._get_gaussian_kernels_on_mesh(
            scalp_downsampled, head_model.scalp.vertices, self.sigma_scalp
        )

        vertices = np.arange(head_model.brain.nvertices + head_model.scalp.nvertices)
        kernel = np.arange(self.G_brain.sizes["kernel"] + self.G_scalp.sizes["kernel"])

        self.G_brain = self.G_brain.assign_coords(
            {
                "vertex": vertices[: head_model.brain.nvertices],
                "kernel": kernel[: self.G_brain.sizes["kernel"]],
            }
        )

        self.G_scalp = self.G_scalp.assign_coords(
            {
                "vertex": vertices[head_model.brain.nvertices:],
                "kernel": kernel[self.G_brain.sizes["kernel"]:],
            }
        )


    def _compute_H(self, Adot : xr.DataArray):
        """Compute the H matrix for spatial basis functions.

        Transforms the sensitivity matrix into the spatial basis space.

        Args:
            Adot: Sensitivity matrix shape=(channel, vertex, wavelength)
        """
        assert Adot.dims == ("channel", "vertex", "wavelength")

        n_channel = len(Adot.channel)
        n_wavelength = len(Adot.wavelength)

        # number of kernels
        n_k_brain = self.G_brain.sizes["kernel"]
        n_k = self.G_brain.sizes["kernel"] + self.G_scalp.sizes["kernel"]

        H = np.zeros((n_channel, n_k, n_wavelength))
        for w in range(n_wavelength):
            Adot_w = Adot.isel(wavelength=w).values
            H[:, :n_k_brain, w] = Adot_w[:, Adot.is_brain] @ self.G_brain.values
            H[:, n_k_brain:, w] = Adot_w[:, ~Adot.is_brain] @ self.G_scalp.values

        is_brain = np.ones(n_k, dtype=np.bool_)
        is_brain[n_k_brain:] = False

        H = xr.DataArray(H, dims=("channel", "kernel", "wavelength"))

        # /!\ the kernel coordinates in H differ from the kernel coordinates
        # in self.G_brain and self.G_scalp
        H = H.assign_coords(
            {
                "channel": Adot.channel,
                "wavelength": Adot.wavelength,
                "kernel": np.concatenate(
                    [self.G_brain.kernel.values, self.G_scalp.kernel.values]
                ),
                "is_brain": ("kernel", is_brain),
            }
        )
        self.H = H


    def kernel_to_image_space_mua(self, X : np.ndarray) -> np.ndarray:
        """Convert kernel space reconstructions to image space for mua.

        Args:
            X: Reconstruction values in kernel space. shape (kernel, ...)

        Returns:
            np.ndarray: Reconstruction values in image space.
        """

        nkernels_brain = self.G_brain.sizes["kernel"]
        has_scalp = X.sizes["kernel"] > nkernels_brain

        sb_X_brain = X[{"kernel":slice(0,nkernels_brain)}]

        X_brain = xrutils.contract(self.G_brain, sb_X_brain, dim="kernel")

        if has_scalp:
            sb_X_scalp = X[{"kernel": slice(nkernels_brain, None)}]
            X_scalp = xrutils.contract(self.G_scalp, sb_X_scalp, dim="kernel")

        if has_scalp:
            is_brain = np.zeros(
                X_brain.sizes["vertex"] + X_scalp.sizes["vertex"], dtype=bool
            )
            is_brain[:X_brain.sizes["vertex"]] = True
            X = xr.concat([X_brain, X_scalp], dim="vertex").assign_coords(
                {"is_brain": ("vertex", is_brain)}
            )
        else:
            is_brain = np.ones(X_brain.sizes["vertex"], dtype=bool)
            X = X_brain.assign_coords({"is_brain": ("vertex", is_brain)})

        return X

        """
        nkernels_brain = self.G_brain.sizes["kernel"]
        has_scalp = (
            len(X) > nkernels_brain if len(X.shape) < 2 else X.shape[0] > nkernels_brain
        )

        if len(X.shape) < 2:
            sb_X_brain = X[:nkernels_brain]
            if has_scalp:
                sb_X_scalp = X[nkernels_brain:]
        else:
            sb_X_brain = X[:nkernels_brain, :]
            if has_scalp:
                sb_X_scalp = X[nkernels_brain:, :]

        # project back to surface space
        X_brain = self.G_brain.values @ sb_X_brain

        if has_scalp:
            X_scalp = self.G_scalp.values @ sb_X_scalp
            # concatenate them back together
            X = np.concatenate([X_brain, X_scalp])
        else:
            # Only brain vertices
            X = X_brain
        return X
        """

    def kernel_to_image_space_conc(self, X) -> np.ndarray:
        """Convert kernel space reconstructions to image space for concentration.

        Args:
            X: Reconstruction values in kernel space.

        Returns:
            np.ndarray: Reconstruction values in image space with HbO/HbR split.
        """

        assert "flat_kernel" in X.dims


        X = xrutils.unstack(X, "flat_kernel", ("chromo", "kernel"))

        # FIXME limited to two chromophores
        #split = X.shape[0]//2
        nkernels_brain = self.G_brain.sizes["kernel"]
        has_scalp = X.sizes["kernel"] > nkernels_brain

        sb_X_brain_hbo = X[{"kernel":slice(0,nkernels_brain)}].loc[{"chromo" : "HbO"}]
        sb_X_brain_hbr = X[{"kernel":slice(0,nkernels_brain)}].loc[{"chromo" : "HbR"}]

        X_hbo_brain = xrutils.contract(self.G_brain, sb_X_brain_hbo, dim="kernel")
        X_hbr_brain = xrutils.contract(self.G_brain, sb_X_brain_hbr, dim="kernel")

        X_brain = xr.concat(
            [X_hbo_brain, X_hbr_brain], dim="chromo", coords={"chromo": ["HbO", "HbR"]}
        )

        if has_scalp:
            sb_X_scalp_hbo = X[{"kernel": slice(nkernels_brain, None)}].loc[
                {"chromo": "HbO"}
            ]
            sb_X_scalp_hbr = X[{"kernel": slice(nkernels_brain, None)}].loc[
                {"chromo": "HbR"}
            ]

            X_hbo_scalp = xrutils.contract(self.G_scalp, sb_X_scalp_hbo, dim="kernel")
            X_hbr_scalp = xrutils.contract(self.G_scalp, sb_X_scalp_hbr, dim="kernel")

            X_scalp = xr.concat([X_hbo_scalp, X_hbr_scalp], dim="chromo")
            X_scalp = X_scalp.assign_coords({"chromo": ["HbO", "HbR"]})
        if has_scalp:
            is_brain = np.zeros(
                X_brain.sizes["vertex"] + X_scalp.sizes["vertex"], dtype=bool
            )
            is_brain[:X_brain.sizes["vertex"]] = True
            X = xr.concat([X_brain, X_scalp], dim="vertex").assign_coords(
                {"is_brain": ("vertex", is_brain)}
            )
        else:
            is_brain = np.ones(X_brain.sizes["vertex"], dtype=bool)
            X = X_brain.assign_coords({"is_brain": ("vertex", is_brain)})

        X = X.transpose("chromo", "vertex", ...)
        return X


        """
        if len(X.shape) > 1:
            X_hbo = X[:split,:]
            X_hbr = X[split:,:]
            sb_X_brain_hbo = X_hbo[:nkernels_brain,:]
            sb_X_brain_hbr = X_hbr[:nkernels_brain,:]

            if has_scalp:
                sb_X_scalp_hbo = X_hbo[nkernels_brain:,:]
                sb_X_scalp_hbr = X_hbr[nkernels_brain:,:]
        else:
            X_hbo = X[:split]
            X_hbr = X[split:]
            sb_X_brain_hbo = X_hbo[:nkernels_brain]
            sb_X_brain_hbr = X_hbr[:nkernels_brain]

            if has_scalp:
                sb_X_scalp_hbo = X_hbo[nkernels_brain:]
                sb_X_scalp_hbr = X_hbr[nkernels_brain:]

        # project back to surface space
        X_hbo_brain = self.G_brain.values @ sb_X_brain_hbo
        X_hbr_brain = self.G_brain.values @ sb_X_brain_hbr

        if has_scalp:
            X_hbo_scalp = self.G_scalp.values @ sb_X_scalp_hbo
            X_hbr_scalp = self.G_scalp.values @ sb_X_scalp_hbr

        # concatenate them back together
        if len(X.shape) == 1:
            if has_scalp:
                X = np.stack(
                    [
                        np.concatenate([X_hbo_brain, X_hbo_scalp]),
                        np.concatenate([X_hbr_brain, X_hbr_scalp]),
                    ],
                    axis=0,
                )
            else:
                X = np.stack([X_hbo_brain, X_hbr_brain], axis=0)
        else:
            if has_scalp:
                X = np.stack(
                    [
                        np.vstack([X_hbo_brain, X_hbo_scalp]),
                        np.vstack([X_hbr_brain, X_hbr_scalp]),
                    ],
                    axis=0,
                )
            else:
                X = np.stack([X_hbo_brain, X_hbr_brain], axis=0)

        return X
        """


    def to_file(self, fname : Path | str):
        """Serialize prepared Gaussian spatial basis functions to HDF5 file.

        Args:
            fname: path of the output file.
        """

        with h5py.File(fname, "w") as fout:
            ioutils.xarray_to_hdfgroup(fout, self.H, "H")
            ioutils.xarray_to_hdfgroup(fout, self.G_brain, "G_brain")
            ioutils.xarray_to_hdfgroup(fout, self.G_scalp, "G_scalp")
            ioutils.xarray_to_hdfgroup(fout, self._mask, "_mask")

            for name in [
                "threshold_brain",
                "threshold_scalp",
                "sigma_brain",
                "sigma_scalp",
                "mask_threshold",
            ]:
                fout["/"].attrs[name] = str(getattr(self, name))



    @classmethod
    def from_file(cls, fname : Path | str) -> "GaussianSpatialBasisFunctions":
        """Load prepared Gaussian spatial basis functions from HDF5 group.

        Args:
            fname: path of the file to read from.

        Returns:
            GaussianSpatialBasisFunctions: Loaded instance.
        """

        sbf = cls.__new__(cls)
        with h5py.File(fname, "r") as f:
            sbf.H = ioutils.xarray_from_hdfgroup(f, "H")
            sbf.G_brain = ioutils.xarray_from_hdfgroup(f, "G_brain")
            sbf.G_scalp = ioutils.xarray_from_hdfgroup(f, "G_scalp")
            sbf._mask = ioutils.xarray_from_hdfgroup(f, "_mask")

            for name in [
                "threshold_brain",
                "threshold_scalp",
                "sigma_brain",
                "sigma_scalp",
            ]:
                setattr(sbf, name, pint.Quantity(f["/"].attrs[name]))

            setattr(sbf, "mask_threshold", float(f["/"].attrs["mask_threshold"]))

        return sbf


#class ParcellationBasisFunctions(SpatialBasisFunctions):
#    pass


# we could define constants of parameters that work well together, e.g. based on
# Laura's parameter scans:
REG_TIKHONOV_ONLY = RegularizationParams(
    alpha_meas=0.001, alpha_spatial=None, apply_c_meas=False
)

REG_TIKHONOV_SPATIAL = RegularizationParams(
    alpha_meas=0.01, alpha_spatial=0.001, apply_c_meas=False
)

SBF_GAUSSIANS_DENSE = dict(
    mask_threshold=-2,
    threshold_brain=1 * units.mm,
    threshold_scalp=5 * units.mm,
    sigma_brain=1 * units.mm,
    sigma_scalp=5 * units.mm,
)

SBF_GAUSSIANS_SPARSE = dict(
    mask_threshold=-2,
    threshold_brain=5 * units.mm,
    threshold_scalp=20 * units.mm,
    sigma_brain=5 * units.mm,
    sigma_scalp=20 * units.mm,
)

# FIXME
# likewise, if there are any heuristics how to set these parameters, we could offer
# functions to compute them
#def estimate_reg_params(*args) -> RegularizationParams:
#    """Estimate regularization parameters from data.
#
#    Args:
#        *args: Variable arguments for parameter estimation.
#
#    Returns:
#        RegularizationParams: Estimated regularization parameters.
#    """
#    pass


class ImageRecon:

    def __init__(
        self,
        Adot,
        *,
        recon_mode: str = "mua",
        brain_only : bool = False,
        regularization_params: RegularizationParams = REG_TIKHONOV_ONLY,
        spatial_basis_functions: SpatialBasisFunctions | None = None,
    ):
        # error handling of invalid params

        self.recon_mode = recon_mode
        self.reg_params = regularization_params
        self.sbf = spatial_basis_functions
        self.Adot = Adot # FIXME remove
        self.brain_only = brain_only

        # cache intermediate matrices to avoid recomputations

        # These would invalidate when Adot or reg./sbf. params. change. Changing
        # these requires a new instance of ImageRecon, so they are considered constants
        # here. Depending on recon_mode they have different shapes.
        self._D = None  # Linv^2 * A.T
        self._F = None  # A_hat * A_hat.T

        # The matrix to transform from absorption changes to concentrations
        self._mua2conc = None

        # this invalidates when C_meas changes
        self._W = None  # the pseudo_inverse (W=D@inv(F+ lambda_meas C))
        self._W_input_hash = None # a hash of C_meas. if C_meas changes, recompute W

        if self.sbf is not None:
            self._prepare(self.sbf.H)
        else:
            self._prepare(self.Adot)

    def reconstruct(
        self,
        y: cdt.NDTimeSeries,
        c_meas: xr.DataArray | None = None,
    ) -> cdt.NDTimeSeries:
        """Reconstruct images from measurement data.

        Args:
            y: optical density time series or time point data.
            c_meas: Measurement covariance matrix (optional).

        Returns:
            cdt.NDTimeSeries: Reconstructed images.
        """

        # y_units = y.pint.units
        y = y.pint.dequantify()

        if (c_meas is None) and self.reg_params.apply_c_meas:
            raise NotImplementedError(
                "c_meas must be provided when apply_c_meas is set."
            )
            # FIXME
            # estimate c_meas from time_series
            #c_meas = ...


        # if (self._W is None): #or (W_input_hash != self._W_input_hash):
        # if c_meas is provided and contains a time dimension, then average over time
        if c_meas is not None:
            time_dim = self._get_time_dimension(c_meas)
            if time_dim is not None:
                c_meas = c_meas.mean(time_dim)

            new_W_input_hash = hashlib.blake2b(
                c_meas.pint.dequantify().values.tobytes()
            ).hexdigest()
        else:
            new_W_input_hash = "no_c_meas"


        # calculate W

        if (self._W_input_hash is None) or (new_W_input_hash != self._W_input_hash):
            self._W_input_hash = new_W_input_hash

            # without spatial regularization:
            if self.reg_params.alpha_spatial is None:
                # with spatial basis functions:
                if self.sbf is not None:
                    if self.recon_mode == "conc":
                        #D = get_stacked_sensitivity(self.sbf.H.sel(
                        # kernel=self.sbf.H.is_brain.values)).T
                        if self.brain_only:
                            D = fwm.ForwardModel.compute_stacked_sensitivity(
                                self.sbf.H.sel(kernel=self.sbf.H.is_brain.values)
                            ).T
                        else:
                            D = fwm.ForwardModel.compute_stacked_sensitivity(
                                self.sbf.H
                            ).T
                    else:
                        if self.brain_only:
                            D = self.sbf.H.sel(kernel=self.sbf.H.is_brain.values)
                        else:
                            D = self.sbf.H
                        D = D.transpose('kernel', 'channel', 'wavelength')
                    self._W = self._get_W(D, c_meas)
                # without spatial basis functions:
                else:
                    # Need to store original Adot for this case
                    if self.recon_mode == "conc":
                        #D = get_stacked_sensitivity(
                        # self.Adot.sel(vertex=self.Adot.is_brain.values)).T
                        if self.brain_only:
                            D = fwm.ForwardModel.compute_stacked_sensitivity(
                                self.Adot.sel(vertex=self.Adot.is_brain.values)
                            ).T
                        else:
                            D = fwm.ForwardModel.compute_stacked_sensitivity(
                                self.Adot
                            ).T
                    else:
                        if self.brain_only:
                            D = self.Adot.sel(vertex=self.Adot.is_brain.values)
                        else:
                            D = self.Adot

                        D = D.transpose('vertex', 'channel', 'wavelength')
                    self._W = self._get_W(D, c_meas)

            # with spatial regularization:
            else:
                self._W = self._get_W(self._D, c_meas)


        if self.recon_mode == "conc":
            #y = y.stack(measurement=("wavelength", "channel")).sortby('wavelength')
            y = fwm.stack_flat_channel(y) # FIXME: move into _get_image_conc
            conc_img = self._get_image_conc(y)
            return conc_img

        mua_img = self._get_image_mua(y)

        if self.recon_mode == "mua":
            return mua_img
        elif self.recon_mode == "mua*mua2conc":
            return xrutils.contract(
                self._mua2conc, mua_img / units.mm, dim=["wavelength"]
            )
        else:
            raise ValueError()  # unreachable


    def get_image_noise(self, c_meas: xr.DataArray):
        """Compute image noise/variance estimates.

        Args:
            c_meas: Measurement covariance matrix.

        Returns:
            xr.DataArray: Image noise estimates.
        """

        if (c_meas is None) and self.reg_params.apply_c_meas:
            # estimate c_meas from time_series
            raise NotImplementedError("c_meas must be provided")

        # calculate hash(c_meas) and (re)compute W if necessary
        # W_input_hash = hash(tuple(c_meas))

        # if (self._W is None): #or (W_input_hash != self._W_input_hash):
        if c_meas is not None:
            time_dim = self._get_time_dimension(c_meas)
            if time_dim is not None:
                c_meas_tmp = c_meas.mean(time_dim)
            else:
                c_meas_tmp = c_meas

        if self.reg_params.alpha_spatial is None:
            if self.sbf is not None:
                if self.recon_mode == "conc":
                    #D = get_stacked_sensitivity(
                    # self.sbf.H.sel(kernel=self.sbf.H.is_brain.values)).T
                    D = fwm.ForwardModel.compute_stacked_sensitivity(
                        self.sbf.H.sel(kernel=self.sbf.H.is_brain.values)
                    ).T
                else:
                    D = self.sbf.H.sel(kernel=self.sbf.H.is_brain.values)
                    D = D.transpose('kernel', 'channel', 'wavelength')
                self._W = self._get_W(D, c_meas_tmp)
            else:
                # Need to store original Adot for this case
                if self.recon_mode == "conc":
                    #D = get_stacked_sensitivity(
                    # self.Adot.sel(vertex=self.Adot.is_brain.values)).T
                    D = fwm.ForwardModel.compute_stacked_sensitivity(
                        self.Adot.sel(vertex=self.Adot.is_brain.values)
                    ).T
                else:
                    D = self.Adot.sel(vertex=self.Adot.is_brain.values)
                    D = D.transpose('vertex', 'channel', 'wavelength')
                self._W = self._get_W(D, c_meas_tmp)
        else:
            self._W = self._get_W(self._D, c_meas_tmp)


            # self._W_input_hash = W_input_hash

        if self.recon_mode == "conc":
            #c_meas = c_meas.stack(
            # measurement=("wavelength", "channel")).sortby('wavelength')
            c_meas = fwm.stack_flat_channel(c_meas)
            conc_img = self._get_image_noise_conc(c_meas)
            return conc_img

        elif self.recon_mode in ["mua", "mua*mua2conc"]:
            mua_img = self._get_image_noise_mua(c_meas)

            if self.recon_mode == "mua":
                return mua_img
            else:
                return self._mua2conc**2 @ mua_img / units.mm**2
        else:
            raise ValueError()  # unreachable


    def get_image_noise_tstat(
        self, time_series: cdt.NDTimeSeries, c_meas: xr.DataArray | None = None
    ):
        """Compute t-statistic images from noise estimates.

        Args:
            time_series: Time series data for statistics computation.
            c_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: T-statistic images.
        """
        # FIXME is this not already images ? so just X_image / X_noise?
        # not sure what time_series and C_meas would be here
        pass

    def to_file(self, fname: Path | str):
        """Serialize to disk."""
        raise NotImplementedError()
        """
        with h5py.File(fname, "w") as f:
            # store params
            # store D,F,W, mua2conc

            sbf_group = f.create_group("sbf")

            if self.sbf:
                self.sbf.to_hdf5_group(sbf_group)
        """


    @classmethod
    def from_file(cls, fname: str | Path) -> "ImageRecon":
        """Load saved instance from disk.

        Args:
            fname: Path to the saved file.

        Returns:
            ImageReco: Loaded ImageReco instance.
        """
        raise NotImplementedError()

    # --- PREPARATION METHODS ---
    def _prepare(self, Adot):
        """Precompute everything that depends only on inputs in the constructor."""
        # FIXME remove:
        if self.reg_params.alpha_spatial is None:
            if self.brain_only:
                Adot = self.Adot.sel(vertex=self.Adot.is_brain.values)

        # calculate D and F for the selected choice of recon_mode and sbf.
        if self.recon_mode == "conc":
            #Adot_stacked = get_stacked_sensitivity(Adot)
            Adot_stacked = fwm.ForwardModel.compute_stacked_sensitivity(Adot)
            self._D, self._F = self._calculate_DF_conc(Adot_stacked)

        elif self.recon_mode in ["mua", "mua*mua2conc"]:
            self._D, self._F = self._calculate_DF_mua(Adot)

        else:
            raise ValueError()  # unreachable

        if self.recon_mode == "mua*mua2conc":
            # calculate _mua2conc # FIXME not sure what this is ?
            E = nirs.get_extinction_coefficients("prahl", Adot.wavelength)
            self._mua2conc = xrutils.pinv(E)


    def _get_W(self, A, C_meas=None):
        """Get the pseudoinverse matrix W for reconstruction.

        Args:
            A: Sensitivity matrix.
            C_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: pseudoinverse matrix W.
        """
        if self.recon_mode == "conc":
            return self._calculate_W_conc(A, C_meas)
        if self.recon_mode in ["mua", "mua*mua2conc"]:
            return self._calculate_W_mua(A, C_meas)


    # --- MATRIX COMPUTATION METHODS ---
    def _calculate_DF(self, A):
        """Calculate intermediate D and F matrices for regularization.

        Args:
            A: Sensitivity matrix.

        Returns:
            D matrix as xr.DataArray
            F matrix as xr.DataArray

        """

        if self.reg_params.alpha_spatial is None:
            dim = A.dims[0]
            F = A.values @ A.values.T
            F_xr = xr.DataArray(F, dims=(f"{dim}_1", f"{dim}_2"))
            D_xr = None
        else:
            B = np.sum((A**2), axis=0)
            b = B.max()

            # GET A_HAT
            lambda_spatial = self.reg_params.alpha_spatial * b

            L = np.sqrt(B + lambda_spatial)
            Linv = 1/L.values
            # Linv = np.diag(Linv)

            A_hat = A * Linv
            dim = A_hat.dims[0]

            #% GET F and D
            F = A_hat.values @ A_hat.values.T
            D = Linv[:, np.newaxis]**2 * A.values.T

            if self.sbf:
                if self.recon_mode in ["mua", "mua*mua2conc"]:
                    vertex_dim = "kernel"
                    channel_dim = "channel"
                else:
                    vertex_dim = "flat_kernel"
                    channel_dim = "flat_channel"
            else:
                if self.recon_mode in ["mua", "mua*mua2conc"]:
                    vertex_dim = "vertex"
                    channel_dim = "channel"
                else:
                    vertex_dim = "flat_vertex"
                    channel_dim = "flat_channel"


            #D_xr = xr.DataArray(D, dims=("flat_vertex", "flat_channel"))
            D_xr = xr.DataArray(D, dims=(vertex_dim, channel_dim))
            D_xr = D_xr.assign_coords(xrutils.coords_from_other(A,dims=D_xr.dims))

            F_xr = xr.DataArray(F, dims=(f"{dim}_1", f"{dim}_2"))

        return D_xr, F_xr


    def _calculate_DF_conc(self, Adot):
        """Calculate D and F matrices for concentration reconstruction.

        Args:
            Adot: Stacked sensitivity matrix for concentration.

        Returns:
            D matrix as xr.DataArray
            F matrix as xr.DataArray
        """
        return self._calculate_DF(Adot)


    def _calculate_DF_mua(self, Adot):
        """Calculate D and F matrices for mua reconstruction.

        Args:
            Adot: Sensitivity matrix with wavelength dimension.

        Returns:
            D matrix as xr.DataArray
            F matrix as xr.DataArray
        """

        D_lst = []
        F_lst = []
        for w in Adot.wavelength:
            D, F = self._calculate_DF(Adot.sel(wavelength=w.values))
            D_lst.append(D)
            F_lst.append(F)

        if all(d is not None for d in D_lst):
            D = xr.concat(D_lst, dim="wavelength")
            D = D.assign_coords({"wavelength": Adot.wavelength})
        else:
            D = None

        F = xr.concat(F_lst, dim="wavelength")
        F = F.assign_coords({"wavelength": Adot.wavelength})

        return D, F


    def _calculate_W(self, A, F, c_meas=None):
        """Calculate pseudoinverse W from sensitivity and regularization.

        Args:
            A: Sensitivity matrix.
            F: Regularization matrix F.
            c_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: pseudoinverse W.
        """
        max_eig = np.max(np.linalg.eigvals(F))
        lambda_meas = self.reg_params.alpha_meas * max_eig

        # A is 2D. Either (vertex x channel) or (kernel x channel)

        if c_meas is not None:
            W = A.values @ np.linalg.inv(F.values + lambda_meas * c_meas)
        else:
            W = A.values @ np.linalg.inv(F.values + lambda_meas * np.eye(A.shape[1]))

        vertex_dim = A.dims[0] # flat_vertex, flat_kernel, kernel
        channel_dim = A.dims[1] # flat_channel, channel

        W_xr = xr.DataArray(W, dims=(vertex_dim, channel_dim))

        #if "parcel" in A.coords:
        #    W_xr = W_xr.assign_coords(
        #        {"parcel": (vertex_dim, A.coords["parcel"].values)}
        #    )
        #if "is_brain" in A.coords:
        #    W_xr = W_xr.assign_coords(
        #        {"is_brain": (vertex_dim, A.coords["is_brain"].values)}
        #    )

        W_xr = W_xr.assign_coords(
            xrutils.coords_from_other(A, dims=W_xr.dims)
        )

        return W_xr

    def _calculate_W_conc(self, A, c_meas=None):
        """Calculate pseudoinverse W for concentration reconstruction.

        Args:
            A: Stacked sensitivity matrix.
            c_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: Pseudoinverse matrix for concentration reconstruction.
        """
        if c_meas is not None:
            #c_meas = c_meas.stack(measurement=("wavelength", "channel")).sortby(
            #    "wavelength"
            #)
            c_meas = fwm.stack_flat_channel(c_meas)
            c_meas = np.diag(c_meas)
        return self._calculate_W(A, self._F, c_meas)

    def _calculate_W_mua(self, A, c_meas=None):
        """Calculate pseudoinverse W for mua reconstruction.

        Args:
            A: Sensitivity matrix with wavelength dimension.
            c_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: Pseudoinverse matrix for mua reconstruction with wavelength
                dimension.
        """
        W = []
        for wavelength in A.wavelength:
            if c_meas is not None:
                c_meas_w = c_meas.sel(wavelength=wavelength)
                c_meas_w = np.diag(c_meas_w)
            else:
                c_meas_w = None

            W_xr = self._calculate_W(
                A.sel(wavelength=wavelength),
                self._F.sel(wavelength=wavelength),
                c_meas_w,
            )
            W.append(W_xr)

        W_xr = xr.concat(W, dim="wavelength")
        W_xr = W_xr.assign_coords({"wavelength": A.wavelength})

        return W_xr

    # --- IMAGE RECONSTRUCTION METHODS ---

    def _get_image_conc(self, y: cdt.NDTimeSeries) -> cdt.NDTimeSeries:
        # Detect time dimension
        time_dim = self._get_time_dimension(y)
        has_time = time_dim is not None
        if has_time:
            y = y.transpose('flat_channel', ...)

        conc_img = self._W @ y

        if self.sbf is None:
            """
            # direct recon without spatial basis
            split = len(self._W.flat_vertex)//2
            if has_time:
                conc_img = conc_img.reshape([2, split, conc_img.shape[1]])
            else:
                conc_img = conc_img.reshape([2, split])
            """
            conc_img = fwm.unstack_flat_vertex(conc_img)
        else:
            # direct recon with spatial basis
            # FIXME: vertex dim name
            conc_img = self.sbf.kernel_to_image_space_conc(conc_img)

        #return self._create_conc_dataarray(conc_img, y, time_dim)
        return conc_img


    def _get_image_mua(self, y):
        """Compute absorption coefficient image from measurements.

        Args:
            y: Optical density measurement data with wavelength dimension.

        Returns:
            xr.DataArray: Absorption coefficient image.
        """

        #time_dim = self._get_time_dimension(y)

        mua_img = xrutils.contract(self._W, y, dim="channel")

        if self.sbf is not None:
            mua_img = self.sbf.kernel_to_image_space_mua(mua_img)

        """
        mua_results = []
        for w in y.wavelength:
            W_w = self._W.sel(wavelength=w)
            y_w = y.sel(wavelength=w)
            X_w = W_w.values @ y_w.values
            if self.sbf is not None:
                X_w = self.sbf.kernel_to_image_space_mua(X_w)

            mua_results.append(X_w)

        # Combine wavelengths: stack along wavelength axis
        mua_img = np.stack(mua_results, axis=0)  # (wavelength, vertex, [time])

        # Create properly formatted DataArray
        return self._create_mua_dataarray(mua_img, y, time_dim)
        """

        return mua_img


    def _get_image_noise_conc(self, c_meas: xr.DataArray | None = None):
        """Compute concentration image noise/variance.

        Args:
            c_meas: Measurement covariance matrix

        Returns:
            xr.DataArray: Image noise with proper dimensions and coordinates
        """
        if c_meas is None:
            raise ValueError("c_meas cannot be None for noise computation")

        # Detect time dimension
        time_dim = self._get_time_dimension(c_meas)
        has_time = time_dim is not None

        # Compute noise variance: diag(W @ C_meas @ W.T)
        if has_time:
            # Vectorized computation over time
            c_meas = c_meas.transpose('flat_channel', time_dim)
            noise_var = self._compute_time_varying_noise(self._W, c_meas)
        else:
            # Single timepoint
            noise_var = self._compute_single_noise(self._W, c_meas)

        # Apply spatial basis transformation if needed
        if self.sbf is not None:
            noise_var = self.sbf.kernel_to_image_space_conc(noise_var)
            # if has_time:
            #     noise_var = noise_var  # Transpose back to (vertex, time)
        else:
            # Reshape for HbO/HbR concentration
            noise_var = self._reshape_conc(noise_var, has_time)

        # Create properly formatted xarray
        return self._create_conc_dataarray(noise_var, c_meas, time_dim)


    def _get_image_noise_mua(self, c_meas: xr.DataArray | None = None):
        """Compute absorption coefficient image noise/variance.

        Args:
            c_meas: Measurement covariance matrix

        Returns:
            xr.DataArray: Image noise with proper dimensions and coordinates
        """
        if c_meas is None:
            raise ValueError("c_meas cannot be None for noise computation")

        # Detect time dimension
        time_dim = self._get_time_dimension(c_meas)
        has_time = time_dim is not None

        noise_var_list = []

        # Process each wavelength separately
        for wavelength in self._W.wavelength:
            W_wl = self._W.sel(wavelength=wavelength)
            c_wl = c_meas.sel(wavelength=wavelength)

            # Compute noise for this wavelength
            if has_time:
                noise_wl = self._compute_time_varying_noise(W_wl, c_wl)
            else:
                noise_wl = self._compute_single_noise(W_wl, c_wl)

            # Apply spatial basis transformation if needed
            if self.sbf is not None:
                noise_wl = self.sbf.kernel_to_image_space_mua(noise_wl)

            noise_var_list.append(noise_wl)

        # Combine wavelengths
        noise_var = np.stack(noise_var_list, axis=0)  # (wavelength, vertex, time)

        # Create properly formatted xarray
        return self._create_mua_dataarray(noise_var, c_meas, time_dim)

    # --- HELPER METHODS FOR IMAGE COMPUTATION ---

    def _get_time_dimension(self, data: xr.DataArray) -> str | None:
        """Detect time dimension in data."""
        for dim in ['time', 'reltime']:
            if dim in data.dims:
                return dim
        return None

    def _compute_single_noise(
        self, W: xr.DataArray, c_meas: xr.DataArray
    ) -> np.ndarray:
        """Compute noise for single timepoint: diag(W @ C @ W.T)."""
        c_diag = np.sqrt(c_meas.values)
        return np.nansum((W.values * c_diag)**2, axis=1)

    def _compute_time_varying_noise(
        self, W: xr.DataArray, c_meas: xr.DataArray
    ) -> np.ndarray:
        """Compute noise for multiple timepoints efficiently.

        This unified method works for both:
        - Full multi-wavelength case (concentration reconstruction)
        - Single wavelength case (mua reconstruction)

        Args:
            W: Weight matrix with dims (vertex, flat_channel) or (vertex, channel)
            c_meas: Covariance with dims (time, flat_channel) or (time, channel)

        Returns:
            np.ndarray: Noise variance with dims (vertex, time)
        """
        # Vectorized computation over time
        c_sqrt = np.sqrt(c_meas.values)  # (time, flat_channel) or (time, channel)
        W_expanded = W.values[:, :, np.newaxis]  # (vertex, flat_channel/channel, 1)

        # Broadcasting: (vertex, flat_channel/channel, time)
        weighted = W_expanded * c_sqrt
        return np.nansum(weighted**2, axis=1)  # (vertex, time)


    def _reshape_conc(self, noise_var: np.ndarray, has_time: bool) -> np.ndarray:
        """Reshape noise variance for concentration (HbO/HbR split).

        Args:
            noise_var: Noise variance array.
            has_time: Whether time dimension is present.

        Returns:
            np.ndarray: Reshaped noise variance with HbO/HbR separation.
        """
        if has_time:
            # Split vertex dimension for HbO/HbR: (vertex, time) -> (2, vertex//2, time)
            n_vertex_half = noise_var.shape[0] // 2
            hbo = noise_var[:n_vertex_half, :]
            hbr = noise_var[n_vertex_half:, :]
            return np.stack([hbo, hbr], axis=0)  # (2, vertex, time)
        else:
            # Split vertex dimension: (vertex,) -> (2, vertex//2)
            n_vertex_half = len(noise_var) // 2
            return noise_var.reshape(2, n_vertex_half)

    def _create_conc_dataarray(
        self, X: np.ndarray, c_meas: xr.DataArray, time_dim: str | None
    ) -> xr.DataArray:
        """Create properly formatted concentration DataArray.

        Args:
            X: Concentration data array.
            c_meas: Measurement data for coordinate extraction.
            time_dim: Name of time dimension (if present).

        Returns:
            xr.DataArray: Formatted concentration DataArray with proper coordinates.
        """
        if time_dim is not None:
            # (2, vertex, time)
            dims = ('chromo', 'vertex', time_dim)
            coords = {
                'chromo': ['HbO', 'HbR'],
                time_dim: c_meas[time_dim],
                'samples': (time_dim, np.arange(len(c_meas[time_dim]))),
                'vertex': np.arange(X.shape[1])
            }
        else:
            dims = ('chromo', 'vertex')
            coords = {"chromo": ["HbO", "HbR"], "vertex": np.arange(X.shape[1])}

        X = xr.DataArray(X, dims=dims, coords=coords)
        return self._add_spatial_coordinates(X)

    def _create_mua_dataarray(
        self, X: np.ndarray, c_meas: xr.DataArray, time_dim: str | None
    ) -> xr.DataArray:
        """Create properly formatted mua DataArray.

        Args:
            X: Mua data array.
            c_meas: Measurement data for coordinate extraction.
            time_dim: Name of time dimension (if present).

        Returns:
            xr.DataArray: Formatted mua DataArray with proper coordinates.
        """
        if time_dim is not None:
            dims = ('wavelength', 'vertex', time_dim)
            coords = {
                'wavelength': c_meas.wavelength,
                time_dim: c_meas[time_dim],
                'samples': (time_dim, np.arange(len(c_meas[time_dim]))),
                'vertex': np.arange(X.shape[1])
            }
        else:
            dims = ('wavelength', 'vertex')
            coords = {"wavelength": c_meas.wavelength, "vertex": np.arange(X.shape[1])}

        noise_da = xr.DataArray(X, dims=dims, coords=coords)
        return self._add_spatial_coordinates(noise_da)


    def _add_spatial_coordinates(self, X: xr.DataArray) -> xr.DataArray:
        """Add parcel and is_brain coordinates if available in Adot."""
        if 'parcel' in self.Adot.coords:
            X = X.assign_coords(
                {"parcel": ("vertex", self.Adot.coords["parcel"].values)}
            )
        if 'is_brain' in self.Adot.coords:
            X = X.assign_coords(
                {"is_brain": ("vertex", self.Adot.coords["is_brain"].values)}
            )
        return X

