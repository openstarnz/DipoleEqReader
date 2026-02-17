import h5py
import numpy as np
import contourpy
import os


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


class DPEqFile:
    def __init__(self, filename):
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
            print(f"Error reading HDF5 file: {e}")
    def traverse(self):
        print_dict_structure(self.data)

    def get_r(self, transpose=False):
        if transpose:
            return self.data["Grid"]["R"]["data"].T
        return self.data["Grid"]["R"]["data"]

    def get_z(self, transpose=False):
        if transpose:
            return self.data["Grid"]["Z"]["data"].T
        return self.data["Grid"]["Z"]["data"]

    def get_psirz(self, transpose=False):
        if transpose:
            return self.data["Grid"]["Psi"]["data"].T
        return self.data["Grid"]["Psi"]["data"]

    def get_Bp_r(self):
        return self.data["Grid"]["Bp_R"]["data"]

    def get_Bp_z(self):
        return self.data["Grid"]["Bp_Z"]["data"]

    def get_inner_wall(self):
        return self.data["Boundaries"]["ilim"]["data"]

    def get_outer_wall(self):
        return self.data["Boundaries"]["olim"]["data"]

    def get_fcfs(self):
        return self.data["Boundaries"]["FCFS"]["data"]

    def get_lcfs(self):
        return self.data["Boundaries"]["LFCS"]["data"]

    def get_z0_idx(self):
        z0 = self.data["Scalars"]["Z0"]["data"]
        Z = self.get_Z()
        return np.argmin(np.abs(Z - z0))

    def get_psi_rmp(self,r):
        if r == 'fcfs':
            return self.data["Scalars"]["PsiFCFS"]["data"]
        elif r == 'lcfs':
            return self.data["Scalars"]["PsiLCFS"]["data"]
        else:
            return np.interp(r, self.get_r(), self.get_psirz()[:, self.get_z0_idx()]) # to be fixed

    def get_psi_contour(self,psi_ref, n_points=1000, closed=False):
        r_psi = self.get_r()
        z_psi = self.get_z()
        psirz = self.get_psirz()
        print("psi_ref", psi_ref)
        contour_generator = contourpy.contour_generator(r_psi, z_psi, psirz, name='serial', total_chunk_count=1)
        if closed:
            return [np.array(c[:-1,:]) for c in contour_generator.contour(psi_ref, n_points=n_points, unique=unique)]
        else:
            return [np.array(c[:-2,:]) for c in contour_generator.lines(psi_ref)]

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

    def _add_psi_interpolator(self):
        from scipy.interpolate import RegularGridInterpolator

        R = self.get_r()
        Z = self.get_z()
        Psirz = self.get_psirz()

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


# # Example usage:
# fn = "/Users/Jerome.Guterl/development/dipole_hermes_mesh/test.h5"

# dpeq = DPEqReader(fn)
# print_dict_structure(dpeq.data)
