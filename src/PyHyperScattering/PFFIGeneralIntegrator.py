import xarray as xr
import numpy as np
import warnings
from pyFAI.integrator.fiber import FiberIntegrator
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator

class PFFIGeneralIntegrator(PFGeneralIntegrator):
    """
    Integrator for GIWAXS/Fiber data using pyFAI's FiberIntegrator.

    Inherits from PFGeneralIntegrator for shared functionality (e.g., mask, geometry).
    """

    def __init__(self,
                 sample_orientation: int = 1,
                 incident_angle: float = 0.12,
                 tilt_angle: float = 0.0,
                 unit_ip: str = "qip_A^-1",
                 unit_oop: str = "qoop_A^-1",
                 npt_ip: int = 500,
                 npt_oop: int = 500,
                 **kwargs):
        self.sample_orientation = sample_orientation
        self.incident_angle = incident_angle
        self.tilt_angle = tilt_angle
        self.unit_ip = unit_ip
        self.unit_oop = unit_oop
        self.npt_ip = npt_ip
        self.npt_oop = npt_oop
        super().__init__(**kwargs)

    def recreateIntegrator(self):
        """Initialize the FiberIntegrator with the geometry parameters."""
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
            detector=None  # Optional if poni file contains this
        )

    def integrateSingleImage(self, da):
        """
        Converts raw fiber GIWAXS detector image to qIP vs qOOP.

        Inputs:
            da: Raw xarray DataArray
        Outputs:
            xarray DataArray of integrated result
        """
        if da.ndim > 2:
            img_to_integ = np.squeeze(da.values)
        else:
            img_to_integ = da.values

        if self.mask is None:
            warnings.warn(f'No mask defined. Creating an empty mask with shape {img_to_integ.shape}', stacklevel=2)
            self.mask = np.zeros_like(img_to_integ)
        assert self.mask.shape == img_to_integ.shape, \
            f"Mask shape {self.mask.shape} does not match image shape {img_to_integ.shape}"

        stacked_axis = [d for d in da.dims if d not in ("pix_x", "pix_y")]
        if stacked_axis:
            stacked_axis = stacked_axis[0]
            system_to_integ = da.indexes[stacked_axis]
        else:
            stacked_axis = "image_num"
            system_to_integ = [0]

        # Perform 2D fiber integration
        result = self.integrator.integrate2d_grazing_incidence(
            data=img_to_integ,
            unit_ip=self.unit_ip,
            unit_oop=self.unit_oop,
            npt_ip=self.npt_ip,
            npt_oop=self.npt_oop,
            mask=self.mask,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle
        )

        out_da = xr.DataArray(
            data=result.intensity,
            dims=["qoop", "qip"],
            coords={
                "qoop": ("qoop", result.oop),
                "qip": ("qip", result.ip)
            },
            attrs=da.attrs
        )

        if stacked_axis in da.coords:
            out_da = out_da.expand_dims(dim={stacked_axis: 1}).assign_coords({stacked_axis: np.array(system_to_integ)})

        return out_da

    def integrate1d(self, da, vertical_integration=True):
        """
        Performs 1D integration over either qIP or qOOP depending on vertical_integration flag.
        """
        img = da.values.squeeze()
        result = self.integrator.integrate1d_grazing_incidence(
            data=img,
            unit_ip=self.unit_ip,
            unit_oop=self.unit_oop,
            npt_ip=self.npt_ip,
            npt_oop=self.npt_oop,
            vertical_integration=vertical_integration,
            mask=self.mask,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle,
        )

        axis = "qip" if vertical_integration else "qoop"
        coord = result.ip if vertical_integration else result.oop
        return xr.DataArray(
            data=result.intensity,
            dims=[axis],
            coords={axis: coord},
            attrs=da.attrs
        )

    def integrate2d_exitangles(self, da):
        """
        Returns 2D integration in horizontal and vertical exit angles.
        """
        img = da.values.squeeze()
        result = self.integrator.integrate2d_exitangles(
            data=img,
            npt_ip=self.npt_ip,
            npt_oop=self.npt_oop,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle,
            mask=self.mask,
        )

        return xr.DataArray(
            data=result.intensity,
            dims=["exit_angle_vertical", "exit_angle_horizontal"],
            coords={
                "exit_angle_vertical": result.oop,
                "exit_angle_horizontal": result.ip
            },
            attrs=da.attrs
        )

    def integrate2d_polar(self, da, polar_degrees=False):
        """
        Returns 2D polar plot: polar angle vs q-modulus.
        """
        img = da.values.squeeze()
        result = self.integrator.integrate2d_polar(
            data=img,
            sample_orientation=self.sample_orientation,
            incident_angle=self.incident_angle,
            tilt_angle=self.tilt_angle,
            polar_degrees=polar_degrees,
            radial_unit="A^-1",
            mask=self.mask,
        )

        return xr.DataArray(
            data=result.intensity,
            dims=["polar_angle", "q_mod"],
            coords={
                "polar_angle": result.azimuthal,
                "q_mod": result.radial
            },
            attrs=da.attrs
        )

    def __str__(self):
        return f"FiberIntegrator wrapper SDD = {self.dist} m, sample_orientation = {self.sample_orientation}, incident_angle = {self.incident_angle} rad"
