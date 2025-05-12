import xarray as xr
import numpy as np
import warnings
from pyFAI.integrator.fiber import FiberIntegrator
from pyFAI.units import get_unit_fiber
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.io.ponifile import PoniFile
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator

## need to define the output space

class PFFIGeneralIntegrator(PFGeneralIntegrator):
    """
    Integrator for GIWAXS/Fiber data using pyFAI's FiberIntegrator.

    Inherits from PFGeneralIntegrator for shared functionality (mask, geometry).
    Provides multiple integration modes (2D scattering, 1D cuts, exit angles, polar).
    All momentum-transfer axes are expressed in reciprocal angstroms (1/Å).
    """

    def __init__(
        self,
        sample_orientation: int = 1,
        incident_angle: float = 0.12,
        tilt_angle: float = 0.0,
        npt_ip: int = 500,
        npt_oop: int = 500,
        **kwargs
    ):
        """
        Parameters
        ----------
        sample_orientation : int, default 1
            Fiber orientation code following EXIF standard (1=normal, 2-8 other rotations).
        incident_angle : float, default 0.12
            Psi angle: grazing-incidence angle towards the beam in radians.
        tilt_angle : float, default 0.0
            Chi angle: sample tilt orthogonal to beam in radians.
        npt_ip : int, default 500
            Number of integration points along the in-plane (q_ip) axis.
        npt_oop : int, default 500
            Number of integration points along the out-of-plane (q_oop) axis.
        **kwargs : dict
            Additional parameters passed to PFGeneralIntegrator (dist, poni1/2, rot1/2/3, pixel1/2, wavelength).
        """
        self.sample_orientation = sample_orientation
        self.incident_angle = incident_angle
        self.tilt_angle = tilt_angle
        self.npt_ip = npt_ip
        self.npt_oop = npt_oop
        super().__init__(**kwargs)

    def recreateIntegrator(self):
        """Instantiate FiberIntegrator with current detector geometry and sample parameters."""
        self.integrator = FiberIntegrator(
            dist=self.dist,
            poni1=self.poni1,
            poni2=self.poni2,
            rot1=self.rot1,
            rot2=self.rot2,
            rot3=self.rot3,
            pixel1=self.pixel1,
            pixel2=self.pixel2,
            wavelength=self.wavelength,
            detector=None
        )

    def integrateSingleImage(self, da: xr.DataArray) -> xr.DataArray:
        """
        Perform 2D grazing-incidence integration on a single detector image.

        Transforms pixel intensities to q_oop (vertical, out-of-plane) vs. q_ip (horizontal, in-plane).
        All units are in reciprocal angstroms (1/Å).

        Parameters
        ----------
        da : xarray.DataArray
            Raw detector image. Must have dims ('pix_y','pix_x') or a stacked axis plus those.

        Returns
        -------
        xarray.DataArray
            2D scattering intensity with dims ('qoop','qip') and a stacked coordinate if present.
        """
        img = np.squeeze(da.values) if da.ndim > 2 else da.values

        if self.mask is None:
            warnings.warn(f"No mask defined, creating empty mask of shape {img.shape}", stacklevel=2)
            self.mask = np.zeros_like(img, dtype=bool)
        if self.mask.shape != img.shape:
            raise ValueError(f"Mask shape {self.mask.shape} does not match image shape {img.shape}")

        stack_dims = [d for d in da.dims if d not in ("pix_x", "pix_y")]
        if stack_dims:
            stack_dim = stack_dims[0]
            coords = da.indexes[stack_dim]
        else:
            stack_dim, coords = None, None

        result = self.integrator.integrate2d_grazing_incidence(
            data=img,
            unit_ip="qip_A^-1",
            unit_oop="qoop_A^-1",
            npt_ip=self.npt_ip,
            npt_oop=self.npt_oop,
            mask=self.mask,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle
        )

        out_da = xr.DataArray(
            data=result.intensity,
            dims=("qoop", "qip"),
            coords={
                "qoop": ("qoop", result.azimuthal, {"units": "1/Å"}),
                "qip": ("qip",     result.radial,    {"units": "1/Å"}),
            },
            attrs=da.attrs
        )
        
        if stack_dim:
            out_da = out_da.expand_dims({stack_dim: len(coords)}) \
                           .assign_coords({stack_dim: coords})

        return out_da

    # def integrate1d(self, da: xr.DataArray, vertical_integration: bool = True) -> xr.DataArray:
    #     """
    #     Perform 1D grazing-incidence integration (line cut).

    #     Chooses between q_ip (vertical integration) or q_oop (horizontal integration).

    #     Parameters
    #     ----------
    #     da : xarray.DataArray
    #         Input image or 2D integrated DataArray.
    #     vertical_integration : bool, default True
    #         If True, integrate along q_oop to get intensity vs q_ip; else along q_ip to get vs q_oop.

    #     Returns
    #     -------
    #     xarray.DataArray
    #         1D intensity vs 'qip' or 'qoop' with units '1/Å'.
    #     """
    #     img = np.squeeze(da.values)
    #     result = self.integrator.integrate1d_grazing_incidence(
    #         data=img,
    #         unit_ip="qip_A^-1",
    #         unit_oop="qoop_A^-1",
    #         npt_ip=self.npt_ip,
    #         npt_oop=self.npt_oop,
    #         vertical_integration=vertical_integration,
    #         mask=self.mask,
    #         sample_orientation=self.sample_orientation,
    #         incident_angle=self.incident_angle,
    #         tilt_angle=self.tilt_angle
    #     )
    #     if vertical_integration:
    #         axis, coord = "qip", result.ip
    #     else:
    #         axis, coord = "qoop", result.oop

    #     return xr.DataArray(
    #         data=result.intensity,
    #         dims=(axis,),
    #         coords={axis: (axis, coord, {"units": "1/Å"})},
    #         attrs=da.attrs
    #     )

    # def integrate2d_exitangles(self, da: xr.DataArray) -> xr.DataArray:
    #     """
    #     Map intensities to detector exit angles (chi/psi).

    #     Provides a 2D intensity map of vertical vs horizontal exit angles in radians.

    #     Parameters
    #     ----------
    #     da : xarray.DataArray
    #         Raw detector image.

    #     Returns
    #     -------
    #     xarray.DataArray
    #         Intensity with dims ('exit_angle_vertical','exit_angle_horizontal') and radian units.
    #     """
    #     img = np.squeeze(da.values)
    #     result = self.integrator.integrate2d_exitangles(
    #         data=img,
    #         npt_ip=self.npt_ip,
    #         npt_oop=self.npt_oop,
    #         sample_orientation=self.sample_orientation,
    #         incident_angle=self.incident_angle,
    #         tilt_angle=self.tilt_angle,
    #         mask=self.mask
    #     )

    #     return xr.DataArray(
    #         data=result.intensity,
    #         dims=("exit_angle_vertical", "exit_angle_horizontal"),
    #         coords={
    #             "exit_angle_vertical": ("exit_angle_vertical", result.oop, {"units": "rad"}),
    #             "exit_angle_horizontal": ("exit_angle_horizontal", result.ip, {"units": "rad"}),
    #         },
    #         attrs=da.attrs
    #     )

    # def integrate2d_polar(self, da: xr.DataArray, polar_degrees: bool = False) -> xr.DataArray:
    #     """
    #     Convert to polar coordinates: azimuthal angle vs q magnitude.

    #     All q values are in 1/Å. Polar angle units default to radians.

    #     Parameters
    #     ----------
    #     da : xarray.DataArray
    #         Raw detector image.
    #     polar_degrees : bool, default False
    #         If True, polar angle coords are in degrees.

    #     Returns
    #     -------
    #     xarray.DataArray
    #         Intensity with dims ('polar_angle','q_mod') and units metadata.
    #     """
    #     img = np.squeeze(da.values)
    #     result = self.integrator.integrate2d_polar(
    #         data=img,
    #         sample_orientation=self.sample_orientation,
    #         incident_angle=self.incident_angle,
    #         tilt_angle=self.tilt_angle,
    #         polar_degrees=polar_degrees,
    #         radial_unit="A^-1",
    #         mask=self.mask
    #     )
    #     angle_units = "deg" if polar_degrees else "rad"
    #     return xr.DataArray(
    #         data=result.intensity,
    #         dims=("polar_angle", "q_mod"),
    #         coords={
    #             "polar_angle": ("polar_angle", result.azimuthal, {"units": angle_units}),
    #             "q_mod": ("q_mod", result.radial, {"units": "1/Å"}),
    #         },
    #         attrs=da.attrs
    #     )

    def __str__(self) -> str:
        return (
            f"FiberIntegrator SDD={self.dist} m, sample_orientation={self.sample_orientation}, "
            f"incident_angle={self.incident_angle} rad, tilt_angle={self.tilt_angle} rad"
        )
