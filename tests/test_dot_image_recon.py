import cedalion.data
import cedalion.dot as dot
import numpy as np
import xarray as xr
import cedalion.xrutils as xrutils
import pytest


@pytest.fixture
def Adot():
    rng = np.random.default_rng(seed=42)

    nchannel = 50
    nvertex = 20
    nwavelength = 2
    Adot = xr.DataArray(
        rng.uniform(0, 20.0, size=(nchannel, nvertex, nwavelength)),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "is_brain": ("vertex", [True] * 15 + [False] * 5),
            "channel": [f"C{c:02d}" for c in range(1, nchannel + 1)],
            "source": ("channel", [f"S{c:02d}" for c in range(1, nchannel + 1)]),
            "detector": ("channel", [f"D{c:02d}" for c in range(1, nchannel + 1)]),
            "wavelength": [760.0, 850.0],
        },
    )

    return Adot


def test_image_recon(Adot):
    rng = np.random.default_rng(seed=42)

    x_true = xr.DataArray(
        np.zeros(Adot.sizes["vertex"]),
        dims=["vertex"],
        coords={"is_brain": ("vertex", Adot.is_brain.values)},
    )

    x_true[5:15] = 1.0

    y = xrutils.contract(Adot, x_true, "vertex")
    # y += rng.normal(0, 2, y.shape)

    recon = dot.ImageRecon(
        Adot,
        recon_mode="mua",
        spatial_basis_functions=None,
        alpha_meas=1e-6,
        alpha_spatial=None,
        apply_c_meas=False,
    )

    x_reco = recon.reconstruct(y)

    assert np.allclose(x_reco[0, :], x_true, atol=1e-4)
