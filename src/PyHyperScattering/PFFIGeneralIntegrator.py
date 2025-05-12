import xarray as xr
import numpy as np
import warnings
from pyFAI.integrator.fiber import FiberIntegrator
from pyFAI.units import get_unit_fiber
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.io.ponifile import PoniFile
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator

class PFFIGeneralIntegrator(PFGeneralIntegrator):
    """
    Integrator for GIWAXS/Fiber data using pyFAI's FiberIntegrator.

    Inherits from PFGeneralIntegrator for shared functionality (mask, geometry).
    Provides multiple integration modes (2D scattering, 1D cuts, exit angles, polar).
    All momentum-transfer axes are expressed in reciprocal angstroms (1/Å).

    Parameters
    ----------
    sample_orientation : int, default 1
        Fiber orientation code following EXIF standard.
    incident_angle : float, default 0.12
        Grazing-incidence angle in degrees (converted to radians).
    tilt_angle : float, default 0.0
        Sample tilt angle in degrees (converted to radians).
    npt_ip : int, default None
        Number of integration points along in-plane (q_ip).
    npt_oop : int, default None
        Number of integration points along out-of-plane (q_oop).
    split_pixels : bool, default True
        If True, use pixel-splitting method for integration; if False, disable splitting (method=None).
    rotate_input : bool, default False
        If True, rotates input image by 90° (swap pix_x, pix_y) before integration.
    **kwargs : dict
        Additional params for PFGeneralIntegrator (dist, poni1/2, rot1/2/3, pixel1/2, wavelength).
    """
    def __init__(
        self,
        sample_orientation: int = 1,
        incident_angle: float = 0.12,
        tilt_angle: float = 0.0,
        npt_ip: int = None,
        npt_oop: int = None,
        split_pixels: bool = False,
        rotate_input: bool = False,
        **kwargs
    ):
        self.sample_orientation = sample_orientation
        self._incident_angle = None
        self._tilt_angle = None
        self.incident_angle = incident_angle
        self.tilt_angle = tilt_angle

        self.npt_ip = npt_ip
        self.npt_oop = npt_oop
        self.rotate_input = rotate_input
        self.split_pixels = split_pixels

        # should we define the output space or integration method?
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

    @property
    def incident_angle(self):
        """Grazing-incidence angle in radians."""
        return self._incident_angle

    @incident_angle.setter
    def incident_angle(self, angle_deg: float):
        """Accepts angle in degrees, stores in radians."""
        self._incident_angle = np.deg2rad(angle_deg)

    @property
    def tilt_angle(self):
        """Tilt angle in radians."""
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, angle_deg: float):
        """Accepts angle in degrees, stores in radians."""
        self._tilt_angle = np.deg2rad(angle_deg)

    def integrateSingleImage(self, da: xr.DataArray) -> xr.DataArray:
        """
        Perform 2D grazing-incidence integration on a single detector image.

        Transforms pixel intensities to q_oop vs. q_ip.
        """
        # raw image array
        img = np.squeeze(da.values) if da.ndim > 2 else da.values
        # original detector dims
        try:
            orig_py = da.sizes["pix_y"]
            orig_px = da.sizes["pix_x"]
        except KeyError:
            raise ValueError("DataArray must have 'pix_y' and 'pix_x' dims.")

        # rotate input if requested
        if self.rotate_input:
            img = np.rot90(img, k=1)
            if self.mask is not None:
                self.mask = np.rot90(self.mask, k=1)

        # mask setup
        if self.mask is None:
            warnings.warn(f"No mask defined, creating empty mask of shape {img.shape}", stacklevel=2)
            self.mask = np.zeros_like(img, dtype=bool)
        if self.mask.shape != img.shape:
            raise ValueError(f"Mask shape {self.mask.shape} does not match image shape {img.shape}")

        # stacked dim
        stack_dims = [d for d in da.dims if d not in ("pix_x", "pix_y")]
        if stack_dims:
            stack_dim = stack_dims[0]
            coords = da.indexes[stack_dim]
        else:
            stack_dim, coords = None, None

        # infer integration grid
        if self.npt_ip is None:
            self.npt_ip = orig_py if self.rotate_input else orig_px
        if self.npt_oop is None:
            self.npt_oop = orig_px if self.rotate_input else orig_py

        # integration
        print(f"Incidence Angle (rad): {self.incident_angle}")
        method = ("bbox", "csr", "cython") if self.split_pixels else 'no'
        result = self.integrator.integrate2d_grazing_incidence(
            data=img,
            unit_ip="qip_A^-1",
            unit_oop="qoop_A^-1",
            npt_ip=self.npt_ip,
            npt_oop=self.npt_oop,
            mask=self.mask,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle,
            method=method
        )

        # wrap into DataArray
        out_da = xr.DataArray(
            data=result.intensity,
            dims=("qoop", "qip"),
            coords={
                "qoop": ("qoop", result.azimuthal, {"units": "1/Å"}),
                "qip":  ("qip",   result.radial,    {"units": "1/Å"}),
            },
            attrs=da.attrs
        )

        if stack_dim:
            out_da = out_da.expand_dims({stack_dim: len(coords)}) \
                           .assign_coords({stack_dim: coords})

        return out_da

    ### Methods to potentially add into the integrateSingleImage workflow provided user options for PFFI integrator.

    def integrate2d_exitangles(self, da: xr.DataArray) -> xr.DataArray:
        """
        Map intensities to detector exit angles (chi/psi).

        Provides a 2D intensity map of vertical vs horizontal exit angles in radians.

        Parameters
        ----------
        da : xarray.DataArray
            Raw detector image.

        Returns
        -------
        xarray.DataArray
            Intensity with dims ('exit_angle_vertical','exit_angle_horizontal') and radian units.
        """
        img = np.squeeze(da.values)
        result = self.integrator.integrate2d_exitangles(
            data=img,
            npt_ip=self.npt_ip,
            npt_oop=self.npt_oop,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle,
            mask=self.mask
        )

        return xr.DataArray(
            data=result.intensity,
            dims=("exit_angle_vertical", "exit_angle_horizontal"),
            coords={
                "exit_angle_vertical": ("exit_angle_vertical", result.oop, {"units": "rad"}),
                "exit_angle_horizontal": ("exit_angle_horizontal", result.ip, {"units": "rad"}),
            },
            attrs=da.attrs
        )

    def integrate2d_polar(self, da: xr.DataArray, polar_degrees: bool = False) -> xr.DataArray:
        """
        Convert to polar coordinates: azimuthal angle vs q magnitude.

        All q values are in 1/Å. Polar angle units default to radians.

        Parameters
        ----------
        da : xarray.DataArray
            Raw detector image.
        polar_degrees : bool, default False
            If True, polar angle coords are in degrees.

        Returns
        -------
        xarray.DataArray
            Intensity with dims ('polar_angle','q_mod') and units metadata.
        """
        img = np.squeeze(da.values)
        result = self.integrator.integrate2d_polar(
            data=img,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle,
            polar_degrees=polar_degrees,
            radial_unit="A^-1",
            mask=self.mask
        )
        angle_units = "deg" if polar_degrees else "rad"
        return xr.DataArray(
            data=result.intensity,
            dims=("polar_angle", "q_mod"),
            coords={
                "polar_angle": ("polar_angle", result.azimuthal, {"units": angle_units}),
                "q_mod": ("q_mod", result.radial, {"units": "1/Å"}),
            },
            attrs=da.attrs
        )

    def integrate1d(self, da: xr.DataArray, vertical_integration: bool = True) -> xr.DataArray:
        """
        Perform 1D grazing-incidence integration (line cut).

        Chooses between q_ip (vertical integration) or q_oop (horizontal integration).

        Parameters
        ----------
        da : xarray.DataArray
            Input image or 2D integrated DataArray.
        vertical_integration : bool, default True
            If True, integrate along q_oop to get intensity vs q_ip; else along q_ip to get vs q_oop.

        Returns
        -------
        xarray.DataArray
            1D intensity vs 'qip' or 'qoop' with units '1/Å'.
        """
        img = np.squeeze(da.values)
        result = self.integrator.integrate1d_grazing_incidence(
            data=img,
            unit_ip="qip_A^-1",
            unit_oop="qoop_A^-1",
            npt_ip=self.npt_ip,
            npt_oop=self.npt_oop,
            vertical_integration=vertical_integration,
            mask=self.mask,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle
        )
        if vertical_integration:
            axis, coord = "qip", result.ip
        else:
            axis, coord = "qoop", result.oop

        return xr.DataArray(
            data=result.intensity,
            dims=(axis,),
            coords={axis: (axis, coord, {"units": "1/Å"})},
            attrs=da.attrs
        )
    
    def __str__(self) -> str:
        return (
            f"FiberIntegrator SDD={self.dist} m, sample_orientation={self.sample_orientation}, "
            f"incident_angle={self.incident_angle} rad, tilt_angle={self.tilt_angle} rad"
        )
