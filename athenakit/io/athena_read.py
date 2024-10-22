import numpy as np
import re
import warnings

### Read files ###

# Read .hst files and return dict of 1D arrays.
def hst(filename, raw=False, strict=False, *args, **kwargs):
    data = {}
    with open(filename, 'r') as data_file:
        line = data_file.readline()
        if (line != '# Athena++ history data\n'):
            raise TypeError(f"Bad file format \"{line.decode('utf-8')}\" ")
        header = data_file.readline()
        data_names = re.findall(r'\[\d+\]=(\S+)', header)
        if len(data_names) == 0:
            raise RuntimeError('Could not parse header')
    # Read data
    try:
        arr=np.loadtxt(filename, *args, **kwargs).T
    except:
        warnings.warn(f"Could not read file {filename} with np.loadtxt, trying hst_complex")
        return hst_complex(filename, raw, *args, **kwargs)
    locs=np.ones(arr[0].shape,dtype=bool)
    if (not raw):
        # Make time monotonic increasing
        mono=np.minimum.accumulate(arr[0][::-1])[::-1]
        if (strict):
            locs=np.append(mono[:-1]<mono[1:],True)
        else:
            tnow=arr[0][-1]
            for i in range(len(arr[0])-1,-1,-1):
                if (arr[0][i] > tnow):
                    locs[i] = False
                else:
                    tnow = arr[0][i]
    # Make dictionary of results
    for i,name in enumerate(data_names):
        data[name]=arr[i][locs]
    return data

# Read .hst files and return dict of 1D arrays.
def hst_complex(filename, raw=False, *args, **kwargs):
    data = {}
    data['time'] = []
    with open(filename, 'r') as data_file:
        lines=data_file.readlines()
        for line in lines:
            if line.startswith('#  '):
                data_names = re.findall(r'\[\d+\]=(\S+)', line)
                for i,name in enumerate(data_names):
                    if name not in data:
                        data[name]=[0.0]*len(data['time'])
            # line is a series of numbers, convert to list of floats
            elif line.startswith(' '):
                data_line = [float(x) for x in line.split()]
                for i,name in enumerate(data_names):
                    data[name].append(data_line[i])
    # convert lists to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    if (not raw):
        # Make time monotonic increasing
        mono=np.minimum.accumulate(data['time'][::-1])[::-1]
        locs=np.append(mono[:-1]<mono[1:],True)
        for key in data:
            data[key]=data[key][locs]
    return data

# Read .tab files and return dict.
def tab(filename):
    # Parse header
    data_dict = {}
    with open(filename, 'r') as data_file:
        line = data_file.readline()
        attributes = re.search(r'time=(\S+)\s+cycle=(\S+)', line)
        line = data_file.readline()
        headings = line.split()[1:]
    data_dict['time'] = float(attributes.group(1))
    data_dict['cycle'] = int(attributes.group(2))

    # Read data
    arr=np.loadtxt(filename).T
    # Make dictionary of results
    for i,name in enumerate(headings):
        data_dict[name]=arr[i]
    return data_dict
