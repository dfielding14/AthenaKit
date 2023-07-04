import h5py

def save_dict_to_hdf5(dic, filename):
    """
    Parameters
    ----------
    dic:
        python dictionary to be converted to hdf5 format
    filename:
        desired name of hdf5 file
    """
    with h5py.File(filename, 'w') as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)

def _recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Parameters
    ----------
    h5file:
        h5py file to be written to
    path:
        path within h5py file to saved dictionary
    dic:
        python dictionary to be converted to hdf5 format
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, np.bool_, str, int, float,
                             bytes, tuple, list)):
            h5file[path + str(key)] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            print('Error: ', key, ":", item)
            raise ValueError('Cannot save %s type' % type(item))

def load_dict_from_hdf5(filename, mode='r'):

    if mode not in ['r', 'r+', 'a+']:
        raise Exception('>>> read mode error')
    with h5py.File(filename, mode) as h5file:
        return _recursively_load_dict_contents_from_group(h5file, '/')

def _recursively_load_dict_contents_from_group(h5file, path):

    dic = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            dic[key] = item[()]
        elif isinstance(item, h5py.Group):
            dic[key] = _recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return dic
