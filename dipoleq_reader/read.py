"""Read DipoleQ HDF5 equilibria and baseline exports.

Example for a native DipoleQ HDF5 file::

    import matplotlib.pyplot as plt
    from dipoleq_reader.read import DPEqFile

    eq = DPEqFile("equilibrium.h5")
    plt.contour(eq.get_r(), eq.get_z(), eq.get_psirz())
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.show()

Native DipoleQ grids are stored in (Z, R) order. Baseline exports store
their grids in (R, Z) order; use ``eq.get_psirz(transpose=True)`` when
plotting a baseline export with R on the horizontal axis.
"""

import h5py
import numpy as np
import contourpy
import os

from scipy.ndimage import zoom
from .plot import DPEqPlotter
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import scipy
def h5_to_dict_recursive(h5_object):
    """
    Recursively convert an HDF5 group or file into a nested dictionary.

    This function iterates through the items in the given HDF5 object (group or file),
    and converts its datasets into numpy arrays and its groups into nested dictionaries.

    Args:
        h5_object (h5py.Group or h5py.File): The HDF5 object to convert.

    Returns:
        dict: A nested dictionary representation of the HDF5 object, where datasets
        are stored as numpy arrays and groups are represented as nested dictionaries.
    """
    result: dict = {}
    for key, item in h5_object.items():
        if isinstance(item, h5py.Dataset):
            # If it's a dataset, get the data as a numpy array
            result[key] = {}
            result[key]["data"] = item[()]
            if "UNITS" in item.attrs.keys():
                result[key]["units"] = str(item.attrs["UNITS"])
            else:
                result[key]["units"] = "N/A"
            if "NAME" in item.attrs.keys():
                result[key]["name"] = str(item.attrs["NAME"])
            else:
                result[key]["name"] = "N/A"

        elif isinstance(item, h5py.Group):
            # If it's a group, recurse into it
            result[key] = h5_to_dict_recursive(item)
    return result


class DPEqFile(DPEqPlotter):
    def __init__(self, filename):
        super().__init__()
        self.data = {}
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File not found: {filename}")
            with h5py.File(filename, "r") as h5f:
                data = h5_to_dict_recursive(h5f)
                for k, v in h5f.attrs.items():
                    data[k.lower()] = v
                data["filename"] = filename
                self.data = data
        except Exception as e:
            raise RuntimeError(f"Error reading dipole equilibrium file {filename}: {e}")

    def traverse(self):
        print_dict_structure(self.data)
        
    def get_psin(self):
        # Native DipoleEq files store psi under FluxFunctions; baseline exports
        # use a one-dimensional psi_1d profile at the file root.
        if "FluxFunctions" in self.data:
            psi = np.asarray(self.data["FluxFunctions"]["psi"]["data"])
            psi_axis, psi_edge = psi[0], psi[-1]
        elif "psi_1d" in self.data:
            psi = np.asarray(self.data["psi_1d"]["data"])
            psi_axis = np.asarray(self.data["simagx"]["data"]).item()
            psi_edge = np.asarray(self.data["sibdry"]["data"]).item()
        else:
            raise KeyError("Could not find a 1D flux profile to compute normalized psi")
        return (psi - psi_axis) / (psi_edge - psi_axis)

    def get_psi_fcfs(self):
        return self.data["Scalars"]["PsiFCFS"]["data"]
    
    def get_rmp_fcfs(self, location="outer"):
        coords = self.get_fcfs()
        r = coords[:, 0]
        if location == "outer":
            return np.max(r)
        elif location == "inner":
            return np.min(r)
    
    def get_rmp_lcfs(self, location="outer"):
        coords = self.get_lcfs()
        r = coords[:, 0]
        if location == "outer":
            return np.max(r)
        elif location == "inner":
            return np.min(r)
    

    
    def get_psi_lcfs(self):
        return self.data["Scalars"]["PsiLCFS"]["data"]
    
    def get_psi_peak(self):
        idx_max = np.argmax(self.data["FluxFunctions"]["ppsi"]["data"])
        return self.data["FluxFunctions"]["psi"]["data"][idx_max]
    
    

    



    def get_r(self, transpose=False, zoom_factor=1 ):
        arr = (self.data["Grid"]["R"]["data"] if "Grid" in self.data
               else self._get_profile("R"))
        arr = zoom(arr, zoom_factor, order=1)
        if transpose:
            return arr.T
        return arr

    def get_z(self, transpose=False, zoom_factor=1):
        arr = (self.data["Grid"]["Z"]["data"] if "Grid" in self.data
               else self._get_profile("Z"))
        arr = zoom(arr, zoom_factor, order=1)
        if transpose:
            return arr.T
        return arr
    
    def get_vol(self, transpose=False, zoom_factor=1):
        arr = self.data["FluxFunctions"]["Vpsi"]["data"]
        arr = zoom(arr, zoom_factor, order=1)
        if transpose:
            return arr.T
        return arr

    

    
    def get_dvdpsi(self, transpose=False, zoom_factor=1):
        arr = self.data["FluxFunctions"]["Vprime"]["data"]
        arr = zoom(arr, zoom_factor, order=1)
        if transpose:
            return arr.T
        return arr
    
    
    def get_dvdpsi_2d(self,transpose=False, divide_by_twopi=False):
        psirz = self.get_psirz(divide_by_twopi=divide_by_twopi)
        dvdpsi_ = self.data["FluxFunctions"]["Vprime"]["data"] 
        psi = self.get_psi_1d(divide_by_twopi=divide_by_twopi)
        dvdpsi_interp = scipy.interpolate.CubicSpline(psi,dvdpsi_)       
        arr = dvdpsi_interp(psirz)
        if transpose:
            return arr.T
        return arr
    
    def get_dvdpsi_interp(self, divide_by_twopi=False):
        psirz = self.get_psirz(divide_by_twopi=divide_by_twopi)
        dvdpsi_ = self.data["FluxFunctions"]["Vprime"]["data"] 
        psi = self.get_psi_1d(divide_by_twopi=divide_by_twopi)
        return scipy.interpolate.CubicSpline(psi,dvdpsi_)
       
    def get_r_1d(self,  zoom_factor=1):
        arr = self.data["FluxFunctions"]["RBetaMax"]["data"]
        arr = zoom(arr, zoom_factor, order=1)
        return arr
    
    def get_psi_1d(self, zoom_factor=1, divide_by_twopi=False):
        arr = (self.data["FluxFunctions"]["psi"]["data"] if "FluxFunctions" in self.data
               else self._get_profile("psi_1d"))
        arr = zoom(arr, zoom_factor, order=1)
        if divide_by_twopi and ("FluxFunctions" in self.data or self._get_cocos() >= 10):
            arr = arr / (2 * np.pi)
        return arr
    
    
    def get_psin_1d(self, zoom_factor=1):
        return zoom(self.get_psin(), zoom_factor, order=1)
    
    def get_p_1d(self, transpose=False, zoom_factor=1):
        arr = self.data["FluxFunctions"]["ppsi"]["data"]
        arr = zoom(arr, zoom_factor, order=1)
        if transpose:
            return arr.T
        return arr

    def _get_profile(self, name):
        """Return a profile dataset from either supported equilibrium layout."""
        # DipoleEq files use named groups, while baseline exports store profiles
        # at the file root. Accept both, case-insensitively.
        for group_name in ("Profiles", "profiles"):
            group = self.data.get(group_name)
            if isinstance(group, dict):
                for key, value in group.items():
                    if key.lower() == name.lower():
                        return np.asarray(value["data"] if isinstance(value, dict) and "data" in value else value)
        for key, value in self.data.items():
            if key.lower() == name.lower() and isinstance(value, dict) and "data" in value:
                return np.asarray(value["data"])
        raise KeyError(f"Profile '{name}' was not found in the equilibrium data")

    def get_ne(self, unit="m^-3"):
        """Get electron density in ``m^-3`` or ``cm^-3``."""
        if unit not in ("m^-3", "cm^-3"):
            raise ValueError("Invalid density unit; expected 'm^-3' or 'cm^-3'.")
        ne = self._get_profile("ne")
        return ne if unit == "m^-3" else ne * 1e-6

    def get_Te(self, unit="eV"):
        """Get the electron temperature profile in eV."""
        if unit not in ("eV", "keV"):
            raise ValueError("Invalid temperature unit; expected 'eV' or 'keV'.")
        te = self._get_profile("Te")
        return te if unit == "eV" else te / 1e3

    def get_Ti(self, unit="eV"):
        """Get the ion temperature profile in eV."""
        if unit not in ("eV", "keV"):
            raise ValueError("Invalid temperature unit; expected 'eV' or 'keV'.")
        ti = self._get_profile("Ti")
        return ti if unit == "eV" else ti / 1e3


# Zoom factor of 2 to create a 4x4 matrix
    def get_psirz(self, transpose=False, zoom_factor=1, divide_by_twopi=False):
        """Return the stored flux grid; transpose only when requested.

        DipoleQ stores (Z, R), while baseline GEQDSK exports store (R, Z).
        """
        arr = (self.data["Grid"]["Psi"]["data"] if "Grid" in self.data
               else self._get_profile("psi"))
        arr = zoom(arr, zoom_factor, order=1)
        if divide_by_twopi and ("Grid" in self.data or self._get_cocos() >= 10):
            arr = arr / (2 * np.pi)
        if transpose:
            return arr.T
        return arr
    
    def get_p_rz(self, transpose=False, zoom_factor=1):
        arr = self.data["Grid"]["Pressure"]["data"]
        arr = zoom(arr, zoom_factor, order=1)
        if transpose:
            return arr.T
        return arr
    
    def get_psi_func(self):
        r = self.get_r()
        z = self.get_z()
        
        return RectBivariateSpline(r, z, self.get_psirz(transpose="Grid" in self.data))
    
    def get_p_func(self):
        r = self.get_r()
        z = self.get_z()
        
        return RectBivariateSpline(r, z, self.get_p_rz(transpose=True))

    def get_Bp_r(self, transpose=False):
        if "Grid" in self.data and "Bp_R" in self.data["Grid"]:
            arr = self.data["Grid"]["Bp_R"]["data"]
        else:
            # Baseline GEQDSK grids are (R, Z).
            psi = self.get_psirz()
            r = self.get_r()
            z = self.get_z()
            dpsi_dz = np.gradient(psi, z, axis=1)
            sigma_bp, flux_factor = self._cocos_bp_convention()
            orientation = 1 if self._get_cocos() % 2 else -1
            arr = orientation * sigma_bp * dpsi_dz / (flux_factor * r[:, np.newaxis])
        return arr.T if transpose else arr

    def get_Bp_z(self, transpose=False):
        if "Grid" in self.data and "Bp_Z" in self.data["Grid"]:
            arr = self.data["Grid"]["Bp_Z"]["data"]
        else:
            psi = self.get_psirz()
            r = self.get_r()
            dpsi_dr = np.gradient(psi, r, axis=0)
            sigma_bp, flux_factor = self._cocos_bp_convention()
            orientation = 1 if self._get_cocos() % 2 else -1
            arr = -orientation * sigma_bp * dpsi_dr / (flux_factor * r[:, np.newaxis])
        return arr.T if transpose else arr

    def _get_cocos(self):
        """Read the COCOS index stored as a root attribute or scalar dataset."""
        value = self.data.get("cocos", self.data.get("COCOS"))
        if value is None:
            raise ValueError("Cannot derive magnetic field: equilibrium has no COCOS metadata")
        return int(np.asarray(value).item())

    def _cocos_bp_convention(self):
        cocos = self._get_cocos()
        index = cocos % 10
        if index not in range(1, 9):
            raise ValueError(f"Unsupported COCOS index {cocos}")
        # COCOS 1-8 store flux per radian; 11-18 store full poloidal flux.
        flux_factor = 2 * np.pi if cocos >= 10 else 1.0
        # Table I of the COCOS convention: sigma_Bp is negative for 3, 4, 7, 8.
        sigma_bp = -1 if index in (3, 4, 7, 8) else 1
        return sigma_bp, flux_factor

    def get_inner_wall(self):
        if "Boundaries" in self.data:
            return self.data["Boundaries"]["ilim"]["data"]
        return np.column_stack((self._get_profile("rlimi"), self._get_profile("zlimi")))

    def get_outer_wall(self):
        if "Boundaries" in self.data:
            return self.data["Boundaries"]["olim"]["data"]
        return np.column_stack((self._get_profile("rlim"), self._get_profile("zlim")))

    def get_fcfs(self):
        if "Boundaries" in self.data:
            return self.data["Boundaries"]["FCFS"]["data"]
        # Baseline exports name the first closed flux surface the inner boundary.
        return np.column_stack((self._get_profile("ribdry"), self._get_profile("zibdry")))

    def get_lcfs(self):
        if "Boundaries" in self.data:
            return self.data["Boundaries"]["LCFS"]["data"]
        # Baseline exports store the LCFS as separate boundary coordinates.
        r = self._get_profile("rbdry")
        z = self._get_profile("zbdry")
        return np.column_stack((r, z))

    def get_z0_idx(self):
        z0 = self.data["Scalars"]["Z0"]["data"]
        Z = self.get_z()
        return np.argmin(np.abs(Z - z0))
    
    def get_z0(self):
        return self.data["Scalars"]["Z0"]["data"]


    def get_psi_rmp(self,r, divide_by_twopi=False):
        if r == 'fcfs':
            if divide_by_twopi:
                return self.data["Scalars"]["PsiFCFS"]["data"] / (2 * np.pi)
            return self.data["Scalars"]["PsiFCFS"]["data"]
        elif r == 'lcfs':
            if divide_by_twopi:
                return self.data["Scalars"]["PsiLCFS"]["data"] / (2 * np.pi)
            return self.data["Scalars"]["PsiLCFS"]["data"]
        else:
            psi_rz = self.get_psirz(transpose="Grid" not in self.data,
                                    divide_by_twopi=divide_by_twopi)
            return np.interp(r, self.get_r(), psi_rz[self.get_z0_idx(), :])

    def get_psi_contour(self,psi_ref, n_points=1000, closed=False, rotate=None, divide_by_twopi=False):
        r_psi = self.get_r()
        z_psi = self.get_z()
        psirz = self.get_psirz(transpose="Grid" not in self.data,
                               divide_by_twopi=divide_by_twopi)
        print("psi_ref", psi_ref)
        contour_generator = contourpy.contour_generator(r_psi, z_psi, psirz, name='serial', total_chunk_count=1)
        if closed:
            arr = [np.array(c[:,:]) for c in contour_generator.lines(psi_ref)]
        else:
            arr = [np.array(c[:-2,:]) for c in contour_generator.lines(psi_ref)]
        # rotate the contour to start at the point closest to the required starting point
        if rotate is not None:
            if rotate == 'omp':
                rotate_point = (np.max(self.get_r()),self.data["Scalars"]["Z0"]["data"])  # point at the OMP, assuming it's at the same R as the O-point
            elif rotate == 'imp':
                rotate_point = (np.min(self.get_r()),self.data["Scalars"]["Z0"]["data"]) # point at the IMP, assuming it's at the same R as the O-point
            else:
                raise ValueError(f"Invalid rotate option: {rotate}")

            arr_rotated = []
            for c in arr:
                dists = np.sqrt((c[:,0] - rotate_point[0])**2 + (c[:,1] - rotate_point[1])**2)
                idx_start = np.argmin(dists)
                print(idx_start, c[idx_start], rotate_point)
                c_rotated = np.roll(c, -idx_start, axis=0)
                arr_rotated.append(c_rotated)
            return arr_rotated
        return arr

    def get_rmin(self):
        return np.min(self.get_r())
    def get_rmax(self):
        return np.max(self.get_r())
    def get_zmin(self):
        return np.min(self.get_z())
    def get_zmax(self):
        return np.max(self.get_z())

    def get_outerwall(self,format='segments'):
        if format == 'r-z':
            olim = self.data["Boundaries"]["olim"]["data"]
            return (olim[:, 1, 0].flatten(), olim[:, 1, 1].flatten())
        elif format == 'segments':
            return self.data["Boundaries"]["olim"]["data"]


    def get_innerwall(self):
        return self.data["Boundaries"]["ilim"]["data"]

    def _add_psi_interpolator(self, divide_by_twopi=False ):
        from scipy.interpolate import RegularGridInterpolator

        R = self.get_r()
        Z = self.get_z()
        Psirz = self.get_psirz(transpose="Grid" in self.data,
                               divide_by_twopi=divide_by_twopi)

        # Create a 2D interpolator for Psi
        self.psi_interpolator = RegularGridInterpolator((R, Z), Psirz)




def traverse(hdf_file):
    """Traverse all datasets in an HDF5 file and print their paths and objects.

    Args:
        hdf_file (str): Path to the HDF5 file to traverse.

    Returns:
        None
    """

    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = "{}/{}".format(prefix, key)
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, "r") as f:
        for path, dset in h5py_dataset_iterator(f):
            print(path, dset)

    return None


def print_dict_structure(d, indent=0):
    """
    Recursively print the structure of a nested dictionary, showing keys,
    and for values, their type and shape if they are numpy arrays.
    """

    for k, v in d.items():
        prefix = "    " * indent + f"- {k}: "
        if isinstance(v, dict):
            if "data" not in v.keys():
                print(prefix)
                print_dict_structure(v, indent + 1)
            else:
                if isinstance(v["data"], np.ndarray):
                    print(
                        prefix
                        + f"ndarray, shape={v['data'].shape}, dtype={v['data'].dtype}"
                        + " [{}]".format(v["units"] if "units" in v.keys() else "N/A")
                    )
                else:
                    print(
                        prefix
                        + f"{type(v['data']).__name__}"
                        + " [{}]".format(v["units"] if "units" in v.keys() else "N/A")
                    )
        else:
            print(prefix + f"{v}")

def read_eqfile(fn):
    return DPEqFile(fn)
