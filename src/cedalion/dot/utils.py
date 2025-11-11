"""Utility functions for image reconstruction."""

import numpy as np
import scipy.stats
import xarray as xr
from scipy.sparse import coo_array

import cedalion
import cedalion.dataclasses as cdc
import cedalion.geometry.segmentation as segm
import cedalion.typing as cdt
from cedalion import xrutils

# FIXME right location?
def map_segmentation_mask_to_surface(
    segmentation_mask: xr.DataArray,
    transform_vox2ras: cdt.AffineTransform,  # FIXME
    surface: cdc.Surface,
    parcels_vox: np.ndarray = None,
    parcels_verts: xr.DataArray = None,
):
    """Find for each voxel the closest vertex on the surface.

    Args:
        segmentation_mask (xr.DataArray): A binary mask of shape (segmentation_type, i,
            j, k).
        transform_vox2ras (xr.DataArray): The affine transformation from voxel to RAS
            space.
        surface (cedalion.dataclasses.Surface): The surface to map the voxels to.
        parcels_vox (np.ndarray, optional): An array of shape (nx, ny, nz) containing
            the parcel label indices for each voxel.
        parcels_verts (xr.DataArray, optional): An array of shape (nvertices,) containing
            the parcellation information for each brain surface vertex.

    Returns:
        coo_array: A sparse matrix of shape (ncells, nvertices) that maps voxels to
            cells.
    """

    assert surface.crs == transform_vox2ras.dims[0]

    cell_coords = segm.cell_coordinates(segmentation_mask, flat=True)
    cell_coords = cell_coords.points.apply_transform(transform_vox2ras)

    cell_coords = cell_coords.pint.to(surface.units).pint.dequantify()

    ncells = cell_coords.sizes["label"]
    nvertices = len(surface.vertices)

    # find indices of cells that belong to the mask
    cell_indices = np.flatnonzero(segmentation_mask.values)

    # for each cell query the closests vertex on the surface
    dists, vertex_indices = surface.kdtree.query(
        cell_coords.values[cell_indices, :], workers=-1
    )

    if parcels_vox is not None and parcels_verts is not None:
        # overwrite voxel labels if in segmentation mask not in brain tissue
        fs_num_labeled_vox = np.sum(np.flatnonzero(parcels_vox))
        parcels_vox *= segmentation_mask.values
        print("Num of labeled voxels before seg-masking: %d\nNum of labeled voxels  after seg-masking: %d" % (fs_num_labeled_vox, np.sum(np.flatnonzero(parcels_vox))))

        # if parcellation is provided, overwrite vertex_indices with mapping
        # constraint to vertices-mapping within the same parcel
        for parcel_id in np.unique(parcels_vox):
            if parcel_id == 0:
                continue
            # get cell indices within this parcel
            parcels_vox_flat = parcels_vox.flatten()
            parcel_cell_indices = np.argwhere(parcels_vox_flat[cell_indices] == parcel_id)[:, 0]
            if len(parcel_cell_indices) == 0:
                continue
            # get vertices within this parcel
            parcel_vertex_indices = np.where(
                parcels_verts.index == parcel_id
            )[0]
            if len(parcel_vertex_indices) == 0:
                continue
            # build a KDTree for the parcel vertices
            from scipy.spatial import KDTree
            parcel_tree = KDTree(surface.vertices[parcel_vertex_indices, :])
            # query the parcel_tree for the parcel_cell_indices
            dists_parcel, vertex_indices_parcel = parcel_tree.query(
                cell_coords.values[parcel_cell_indices, :], workers=-1
            )
            # map back to global vertex indices
            global_vertex_indices_parcel = parcel_vertex_indices[vertex_indices_parcel]
            # update vertex_indices for these cell indices

            vertex_indices[parcel_cell_indices] = global_vertex_indices_parcel

    # construct a sparse matrix of shape (ncells, nvertices)
    # that maps voxels to cells
    map_voxel_to_vertex = coo_array(
        (np.ones(len(cell_indices)), (cell_indices, vertex_indices)),
        shape=(ncells, nvertices),
    )

    return map_voxel_to_vertex


def normal_hrf(t, t_peak, t_std, vmax):
    """Create a normal hrf.

    Args:
        t (np.ndarray): The time points.
        t_peak (float): The peak time.
        t_std (float): The standard deviation.
        vmax (float): The maximum value of the HRF.

    Returns:
        np.ndarray: The HRF.
    """
    hrf = scipy.stats.norm.pdf(t, loc=t_peak, scale=t_std)
    hrf *= vmax / hrf.max()
    return hrf


def create_mock_activation_below_point(
    head_model: "cedalion.dot.TwoSurfaceHeadModel",
    point: cdt.LabeledPoints,
    time_length: cdt.QTime,
    sampling_rate: cdt.QFrequency,
    spatial_size: cdt.QLength,
    vmax: float,
):
    """Create a mock activation below a point.

    Args:
        head_model: The head model.
        point: The point below which to create the activation.
        time_length: The length of the activation.
        sampling_rate: The sampling rate.
        spatial_size: The spatial size of the activation.
        vmax: The maximum value of the activation.

    Returns:
        xr.DataArray: The activation.
    """
    # assert head_model.crs == point.points.crs

    _, vidx = head_model.brain.kdtree.query(point)

    # FIXME for simplicity use the euclidean distance here whilw the geodesic distance
    # would be the correct choice
    dists = xrutils.norm(
        head_model.brain.vertices - head_model.brain.vertices[vidx, :],
        head_model.brain.crs,
    )

    nsamples = int((time_length * sampling_rate).to_reduced_units().magnitude.item())
    t = np.arange(nsamples) / sampling_rate

    func_spat = np.exp(-((dists / spatial_size) ** 2)).rename({"label": "vertex"})
    func_temp = xr.DataArray(normal_hrf(t, 10, 3, vmax), dims="time")

    activation = func_temp * func_spat
    activation = activation.assign_coords({"time": t})
    return activation
