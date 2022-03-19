import os
import numpy as np
import struct
import h5py
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from bin_convert.py
def read_binary(filename):
    """
    Reads a bin file from filename to dictionary.

    args:
      filename - string
          filename of bin file to read

    returns:
      filedata - dict
          dictionary of fluid file data
    """

    filedata = {}

    # load file and get size
    fp = open(filename, 'rb')
    fp.seek(0, 2)
    filesize = fp.tell()
    fp.seek(0, 0)

    # load header information and validate file format
    code_header = fp.readline().split()
    if len(code_header) < 1:
        raise TypeError("unknown file format")
    if code_header[0] != b"Athena":
        raise TypeError(f"bad file format \"{code_header[0].decode('utf-8')}\" " +
                        "(should be \"Athena\")")
    version = code_header[-1].split(b'=')[-1]
    if version != b"1.1":
        raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

    pheader_count = int(fp.readline().split(b'=')[-1])
    pheader = {}
    for _ in range(pheader_count-1):
        key, val = [x.strip() for x in fp.readline().decode('utf-8').split('=')]
        pheader[key] = val
    time = float(pheader['time'])
    cycle = int(pheader['cycle'])
    locsizebytes = int(pheader['size of location'])
    varsizebytes = int(pheader['size of variable'])

    nvars = int(fp.readline().split(b'=')[-1])
    var_list = [v.decode('utf-8') for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b'=')[-1])
    header = [line.decode('utf-8').split('#')[0].strip()
              for line in fp.read(header_size).split(b'\n')]
    header = [line for line in header if len(line) > 0]

    if locsizebytes not in [4, 8]:
        raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
    if varsizebytes not in [4, 8]:
        raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

    locfmt = 'd' if locsizebytes == 8 else 'f'
    varfmt = 'd' if varsizebytes == 8 else 'f'

    # load grid information from header and validate
    def get_from_header(header, blockname, keyname):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if not blockname.startswith('<'):
            blockname = '<' + blockname
        if blockname[-1] != '>':
            blockname += '>'
        block = '<none>'
        for line in [entry for entry in header]:
            if line.startswith('<'):
                block = line
                continue
            key, value = line.split('=')
            if block == blockname and key.strip() == keyname:
                return value
        raise KeyError(f'no parameter called {blockname}/{keyname}')

    Nx1 = int(get_from_header(header, '<mesh>', 'nx1'))
    Nx2 = int(get_from_header(header, '<mesh>', 'nx2'))
    Nx3 = int(get_from_header(header, '<mesh>', 'nx3'))
    nx1 = int(get_from_header(header, '<meshblock>', 'nx1'))
    nx2 = int(get_from_header(header, '<meshblock>', 'nx2'))
    nx3 = int(get_from_header(header, '<meshblock>', 'nx3'))

    x1min = float(get_from_header(header, '<mesh>', 'x1min'))
    x1max = float(get_from_header(header, '<mesh>', 'x1max'))
    x2min = float(get_from_header(header, '<mesh>', 'x2min'))
    x2max = float(get_from_header(header, '<mesh>', 'x2max'))
    x3min = float(get_from_header(header, '<mesh>', 'x3min'))
    x3max = float(get_from_header(header, '<mesh>', 'x3max'))

    # load data from each meshblock
    n_vars = len(var_list)
    mb_fstr = f"={nx1*nx2*nx3*n_vars}" + varfmt
    mb_varsize = varsizebytes*nx1*nx2*nx3*n_vars
    mb_count = 0

    mb_logical = []
    mb_geometry = []

    mb_data = {}
    for var in var_list:
        mb_data[var] = []

    while fp.tell() < filesize:
        mb_count += 1

        mb_logical.append(np.array(struct.unpack('@4i', fp.read(16))))
        mb_geometry.append(np.array(struct.unpack('=6'+locfmt,
                                    fp.read(6*locsizebytes))))

        data = np.array(struct.unpack(mb_fstr, fp.read(mb_varsize)))
        data = data.reshape(nvars, nx3, nx2, nx1)
        for vari, var in enumerate(var_list):
            mb_data[var].append(data[vari])

    fp.close()

    filedata['time'] = time
    filedata['cycle'] = cycle
    filedata['var_names'] = var_list

    filedata['Nx1'] = Nx1
    filedata['Nx2'] = Nx2
    filedata['Nx3'] = Nx3
    filedata['nvars'] = nvars

    filedata['x1min'] = x1min
    filedata['x1max'] = x1max
    filedata['x2min'] = x2min
    filedata['x2max'] = x2max
    filedata['x3min'] = x3min
    filedata['x3max'] = x3max

    filedata['n_mbs'] = mb_count
    filedata['nx1_mb'] = nx1
    filedata['nx2_mb'] = nx2
    filedata['nx3_mb'] = nx3

    filedata['mb_logical'] = np.array(mb_logical)
    filedata['mb_geometry'] = np.array(mb_geometry)
    filedata['mb_data'] = mb_data

    filedata['header'] = header

    return filedata

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
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, int, float,
                             bytes, tuple, list)):
            h5file[path + str(key)] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            print('Error: ', item)
            raise ValueError('Cannot save %s type' % type(item))


def load_hdf5_to_dict(h5file, path):
    """
    Parameters
    ----------
    h5file:
        h5py file to be loaded as a dictionary
    path:
        path within h5py file to load: '/' for the whole h5py file

    Returns
    -------
    dic:
        dictionary with hdf5 file group content
    """
    dic = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            dic[key] = item[()]
        elif isinstance(item, h5py.Group):
            dic[key] = load_hdf5_to_dict(h5file, path + key + '/')
        else:
            raise ValueError('Cannot load %s type' % type(item))
    return dic

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


class AthenaBinaries:
    def __init__(self,binarypath,path='',label='',name='Base',nlim=1001,update=True,**kwargs):
        self.nlim=nlim
        self.alist=[]
        self.abins=[None,]*self.nlim
        self.binarypath=binarypath
        self.path=path
        self.label=label
        self.name=name
        self.update=update
        return

    def read(self,ilist,info=False):
        for i in ilist:
            if(self.update or i not in self.alist):
                if(info): print("read:",i)
                self.abins[i]=AthenaBinary(path=self.path,label=self.label,num=i)
                self.abins[i].load_binary(self.binarypath+f"{i:05d}.bin")
            self.alist=sorted(list(set(self.alist + list([i]))))
        return

    def save_hdf5(self,ilist=None,hdf5path=None,info=False):
        ilist=self.alist if not ilist else ilist
        hdf5path="../simu/"+self.path+self.label+f"/data/hdf5/" if not hdf5path else hdf5path
        if not os.path.isdir(hdf5path):
            os.mkdir(hdf5path)
        for i in ilist:
            if(info): print("save hdf5:",i)
            abin=self.abins[i]
            abin.save_hdf5(hdf5path+f"{self.name}.{i:05d}.hdf5")
    
    def load_hdf5(self,ilist,hdf5path=None,info=False):
        hdf5path="../simu/"+self.path+self.label+f"/data/hdf5/" if not hdf5path else hdf5path
        for i in ilist:
            if(self.update or i not in self.alist):
                if(info): print("load hdf5:",i)
                self.abins[i]=AthenaBinary(path=self.path,label=self.label,num=i)
                self.abins[i].load_hdf5(hdf5path+f"{self.name}.{i:05d}.hdf5")
            self.alist=sorted(list(set(self.alist + list([i]))))
        self.abin=self.abins[self.alist[0]]
        return

    def config(self,**kwargs):
        self.abin=self.abins[self.alist[0]]
        if(self.update or not self.abin.config_flag):
            self.abin.config(**kwargs)
        for i in self.alist[1:]:
            # only set coord once
            abin=self.abins[i]
            if(self.update or not abin.coord):
                abin.coord=self.abin.coord
            if(self.update or not abin.config_flag):
                abin.config(**kwargs)
        return

    def get_radial(self,ilist=[],info=False,update=False,**kwargs):
        for i in (self.alist if not ilist else ilist):
            if(update or self.update or not self.abins[i].rad):
                if(info): print("get_radial:",i)
                self.abins[i].get_radial(update=update,**kwargs)
        return

    def _remove(self,i):
        if i in self.alist:
            self.alist.remove(i)
            abin=self.abins[i]
            del abin
            self.abins[i]=None
        return

    def remove(self,*args):
        if isinstance(args[0],list):
            for i in args[0]:
                self._remove(i)
        else:
            for i in args:
                self._remove(i)
        return

    def clear(self):
        self.alist=[]
        self.abins=[None,]*self.nlim
        return

    def plot_snapshot(self,ilist,figpath=None,info=False,**kwargs):
        figpath="../figure/Simu_"+self.path+self.label+f"/" if not figpath else figpath
        if not os.path.isdir(figpath):
            os.mkdir(figpath)
        for i in ilist:
            if (info): print("plot snapshot:",i)
            fig=self.abins[i].plot_snapshot(**kwargs)
        return fig

    def make_movie(self,varname,range,duration=0.05,fps=24):
        from moviepy.editor import ImageClip, concatenate_videoclips
        img = (f"../figure/Simu_{self.path}{self.label}/fig_{varname}_{i:04d}.png" for i in range)
        clips = [ImageClip(m).set_duration(duration) for m in img]
        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(f"../figure/Simu_{self.path}{self.label}/video_{varname}.mp4",fps=fps)
        return

class AthenaBinary:
    def __init__(self,path='',label='',num=0):
        self.path=path
        self.label=label
        self.num=num
        self._header={}
        self.raw={}
        self.coord={}
        self.rad={}
        self.slice={}
        self.dist={}
        self.dist2d={}
        return

    def load_binary(self,filename):
        self._init_from_raw(read_binary(filename))

    def _init_from_raw(self,raw):
        self.raw=raw
        #(['time', 'cycle', 'var_names', 'Nx1', 'Nx2', 'Nx3', 'nvars', 'x1min', 'x1max',
        # 'x2min', 'x2max', 'x3min', 'x3max', 'n_mbs', 'nx1_mb', 'nx2_mb', 'nx3_mb', 
        # 'mb_logical', 'mb_geometry', 'mb_data'])
        self._init_header(self.raw['header'])
        self.time=self.raw['time']
        self.cycle=self.raw['cycle']
        self.Nx1=self.raw['Nx1']
        self.Nx2=self.raw['Nx2']
        self.Nx3=self.raw['Nx3']
        self.nvars=self.raw['nvars']
        self.x1min=self.raw['x1min']
        self.x1max=self.raw['x1max']
        self.x2min=self.raw['x2min']
        self.x2max=self.raw['x2max']
        self.x3min=self.raw['x3min']
        self.x3max=self.raw['x3max']
        self.n_mbs=self.raw['n_mbs']
        self.nx1_mb=self.raw['nx1_mb']
        self.nx2_mb=self.raw['nx2_mb']
        self.nx3_mb=self.raw['nx3_mb']
        self.mb_logical=np.asarray(self.raw['mb_logical'])
        self.mb_geometry=np.asarray(self.raw['mb_geometry'])
        self.mb_data=self.raw['mb_data']
        for var in self.raw['var_names']:
            self.mb_data[var]=np.asarray(self.mb_data[var])
        self.use_e=float(self.header('hydro','use_e',True))
        self.gamma=float(self.header('hydro','gamma',5/3))
        self.init_flag=False
        self.config_flag=False
        self.rmin = None
        self.rmax = None
        return
    
    def save_hdf5(self,filename):
        dic={}
        for k,v in self.__dict__.items():
            if (k in ['raw', 'mb_logical', 'mb_geometry', 'mb_data', 'coord']):
                continue
            dic[k]=v
        save_dict_to_hdf5(dic,filename)
        return

    def load_hdf5(self,filename):
        self._init_from_dic(load_dict_from_hdf5(filename))
        return

    def _init_from_dic(self,dic):
        for k,v in dic.items():
            self.__dict__[k]=v
        return

    def init(self):
        if (not self.init_flag):
            if (not self.coord): self.set_coord()
            self.rmin = np.min(self.coord['r'].min())
            self.rmax = np.min(np.abs([self.x1min,self.x1max,self.x2min,self.x2max,self.x3min,self.x3max]))
            self.init_flag=True
        return

    def config(self):
        if (not self.config_flag):
            self.init()
            self.config_data()
            self.config_flag=True
        return

    def _init_header(self, header):
        for line in [entry for entry in header]:
            if line.startswith('<'):
                block = line
                self._header[block]={}
                continue
            key, value = line.split('=')
            self._header[block][key.strip()] = value

    def header(self, blockname, keyname, default=None):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if not blockname.startswith('<'):
            blockname = '<' + blockname
        if blockname[-1] != '>':
            blockname += '>'
        if blockname in self._header.keys():
            if keyname in self._header[blockname].keys():
                return self._header[blockname][keyname]
        print(f'no parameter called {blockname}/{keyname}, return default value')
        return default

    def set_coord(self):
        x=np.swapaxes(np.linspace(self.mb_geometry[:,0],self.mb_geometry[:,0]+(self.nx1_mb-1)*self.mb_geometry[:,3],self.nx1_mb),0,1)
        y=np.swapaxes(np.linspace(self.mb_geometry[:,1],self.mb_geometry[:,1]+(self.nx2_mb-1)*self.mb_geometry[:,4],self.nx2_mb),0,1)
        z=np.swapaxes(np.linspace(self.mb_geometry[:,2],self.mb_geometry[:,2]+(self.nx3_mb-1)*self.mb_geometry[:,5],self.nx3_mb),0,1)
        ZYX=np.swapaxes(np.asarray([np.meshgrid(z[i],y[i],x[i]) for i in range(self.n_mbs)]),0,1)
        self.coord['x'],self.coord['y'],self.coord['z']=ZYX[2].swapaxes(1,2),ZYX[1].swapaxes(1,2),ZYX[0].swapaxes(1,2)
        self.coord['r']=np.sqrt(self.coord['x']**2+self.coord['y']**2+self.coord['z']**2)
        dx=np.asarray([np.full((self.nx3_mb,self.nx2_mb,self.nx1_mb),self.mb_geometry[i,3]) for i in range(self.n_mbs)])
        dy=np.asarray([np.full((self.nx3_mb,self.nx2_mb,self.nx1_mb),self.mb_geometry[i,4]) for i in range(self.n_mbs)])
        dz=np.asarray([np.full((self.nx3_mb,self.nx2_mb,self.nx1_mb),self.mb_geometry[i,5]) for i in range(self.n_mbs)])
        self.coord['dx'],self.coord['dy'],self.coord['dz']=dx,dy,dz
        self.coord['vol']=self.coord['dx']*self.coord['dy']*self.coord['dz']
        return
    
    def config_data(self):
        if (self.use_e):
            self.mb_data['temp']=(self.gamma-1)*self.mb_data['eint']/self.mb_data['dens']
        else:
            self.mb_data['temp']=self.mb_data['eint']
            self.mb_data['eint']=self.mb_data['temp']*self.mb_data['dens']/(self.gamma-1)
        self.mb_data['velr']=(self.mb_data['velx']*self.coord['x']+\
                              self.mb_data['vely']*self.coord['y']+\
                              self.mb_data['velz']*self.coord['z'])/self.coord['r']
        return

    def data(self,var):
        if (var in ['x','y','z','r','dx','dy','dz','vol']):
            return self.coord[var]
        elif (var in ['dens','velx','vely','velz','velr','temp','eint']):
            return self.mb_data[var]
        elif (var in ['mass','pres','entropy','momr','velin','velout','amx','amy','amz',\
            'ekin','etot','mdot','mdotin','mdotout']):
            if (var=='mass'):
                return self.coord['vol']*self.mb_data['dens']
            elif (var=='pres'):
                return (self.gamma-1)*self.mb_data['eint']
            elif (var=='entropy'):
                return self.data('pres')/self.mb_data['dens']**self.gamma
            elif (var=='momr'):
                return self.mb_data['velr']*self.mb_data['dens']
            elif (var=='velin'):
                return np.minimum(self.mb_data['velr'],0.0)
            elif (var=='velout'):
                return np.maximum(self.mb_data['velr'],0.0)
            elif (var=='amx'):
                return self.data('y')*self.data('velz')-self.data('z')*self.data('vely')
            elif (var=='amy'):
                return self.data('z')*self.data('velx')-self.data('x')*self.data('velz')
            elif (var=='amz'):
                return self.data('x')*self.data('vely')-self.data('y')*self.data('velx')
            elif (var=='vtot'):
                return self.data('velx')**2+self.data('velx')**2+self.data('velx')**2
            elif (var=='ekin'):
                return 0.5*self.mb_data['dens']*self.data('vtot')
            elif (var=='etot'):
                return self.data('ekin')+self.mb_data['eint']
            elif (var=='mdot'):
                return self.mb_data['dens']*self.mb_data['velr']
            elif (var=='mdotin'):
                return self.mb_data['dens']*self.data('velin')
            elif (var=='mdotout'):
                return self.mb_data['dens']*self.data('velout')
            else:
                print(f"ERROR: No data callled {var}!!!")
        else:
            print(f"ERROR: No data callled {var}!!!")
        return None

    def get_data(self,var,level=0,xyz=[]):
        if (not xyz):
            xyz = [self.x1min,self.x1max,self.x2min,self.x2max,self.x3min,self.x3max]
        root_level = self.mb_logical[:,-1].min()
        max_level = self.mb_logical[:,-1].max()
        logical_level = level+root_level
        nx1_fac = 2**level*self.Nx1/(self.x1max-self.x1min)
        nx2_fac = 2**level*self.Nx2/(self.x2max-self.x2min)
        nx3_fac = 2**level*self.Nx3/(self.x3max-self.x3min)
        i_min = int((xyz[0]-self.x1min)*nx1_fac)
        i_max = int((xyz[1]-self.x1min)*nx1_fac)
        j_min = int((xyz[2]-self.x2min)*nx2_fac)
        j_max = int((xyz[3]-self.x2min)*nx2_fac)
        k_min = int((xyz[4]-self.x3min)*nx3_fac)
        k_max = int((xyz[5]-self.x3min)*nx3_fac)
        data = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min))
        raw = self.data(var)
        for nmb in range(self.n_mbs):
            block_level = self.mb_logical[nmb,-1]
            block_loc = self.mb_logical[nmb,:3]
            block_data = raw[nmb]
            
            # Prolongate coarse data and copy same-level data
            if (block_level <= logical_level):
                s = int(2**(logical_level - block_level))
                # Calculate destination indices, without selection
                il_d = block_loc[0] * self.nx1_mb * s if self.Nx1 > 1 else 0
                jl_d = block_loc[1] * self.nx2_mb * s if self.Nx2 > 1 else 0
                kl_d = block_loc[2] * self.nx3_mb * s if self.Nx3 > 1 else 0
                iu_d = il_d + self.nx1_mb * s if self.Nx1 > 1 else 1
                ju_d = jl_d + self.nx2_mb * s if self.Nx2 > 1 else 1
                ku_d = kl_d + self.nx3_mb * s if self.Nx3 > 1 else 1
                # Calculate (prolongated) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue
                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min
                if s > 1:
                    if self.Nx1 > 1:
                        block_data = np.repeat(block_data, s, axis=2)
                    if self.Nx2 > 1:
                        block_data = np.repeat(block_data, s, axis=1)
                    if self.Nx3 > 1:
                        block_data = np.repeat(block_data, s, axis=0)
                data[kl_d:ku_d,jl_d:ju_d,il_d:iu_d]=block_data[kl_s:ku_s,jl_s:ju_s,il_s:iu_s]
            # Restrict fine data, volume average
            else:
                # Calculate scale
                s = int(2 ** (block_level - logical_level))
                # Calculate destination indices, without selection
                il_d = int(block_loc[0] * self.nx1_mb / s) if self.Nx1 > 1 else 0
                jl_d = int(block_loc[1] * self.nx2_mb / s) if self.Nx2 > 1 else 0
                kl_d = int(block_loc[2] * self.nx3_mb / s) if self.Nx3 > 1 else 0
                iu_d = int(il_d + self.nx1_mb / s) if self.Nx1 > 1 else 1
                ju_d = int(jl_d + self.nx2_mb / s) if self.Nx2 > 1 else 1
                ku_d = int(kl_d + self.nx3_mb / s) if self.Nx3 > 1 else 1
                #print(kl_d,ku_d,jl_d,ju_d,il_d,iu_d)
                # Calculate (restricted) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue
                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min
                
                # Account for restriction in source indices
                num_extended_dims = 0
                if self.Nx1 > 1:
                    il_s *= s
                    iu_s *= s
                    num_extended_dims += 1
                if self.Nx2 > 1:
                    jl_s *= s
                    ju_s *= s
                    num_extended_dims += 1
                if self.Nx3 > 1:
                    kl_s *= s
                    ku_s *= s
                    num_extended_dims += 1
                
                # Calculate fine-level offsets
                io_vals = range(s) if self.Nx1 > 1 else (0,)
                jo_vals = range(s) if self.Nx2 > 1 else (0,)
                ko_vals = range(s) if self.Nx3 > 1 else (0,)

                # Assign values
                for ko in ko_vals:
                    for jo in jo_vals:
                        for io in io_vals:
                            data[kl_d:ku_d,jl_d:ju_d,il_d:iu_d] += block_data[
                                                                kl_s+ko:ku_s:s,
                                                                jl_s+jo:ju_s:s,
                                                                il_s+io:iu_s:s]\
                                                                /(s**num_extended_dims)
        return data

    def set_radial(self,varl=['dens','temp','velr','mdot'],varsuf='',bins=1000,\
        locs=None,weights='vol',update=False):
        for var in varl:
            varname = var+varsuf
            if (update or varname not in self.rad.keys()):
                logr = np.log10(self.coord['r'])
                weinorm = self.data(weights)
                if(locs is not None):
                    weinorm=weinorm*locs
                hist = np.histogram(logr,bins=bins,weights=weinorm)
                self.rad['r'] = (10**((hist[1][:-1]+hist[1][1:])/2))
                rmax = self.rmax
                rlocs = np.logical_and(hist[0]!=0,self.rad['r']<rmax)
                self.rad['r'] = self.rad['r'][rlocs]
                drlocs = np.append(rlocs,True)
                self.rad['dr'] = (10**((hist[1][drlocs])[1:])-10**((hist[1][drlocs])[:-1]))
                norm = hist[0][rlocs]
                break
        for var in varl:
            varname = var+varsuf
            if (update or varname not in self.rad.keys()):
                dat = np.histogram(logr,bins=bins,weights=weinorm*self.data(var))
                if (var in ['dens','velx','vely','velz','velr','temp','eint']):
                    self.rad[varname] = dat[0][rlocs]/norm
                elif (var in ['pres','entropy','momr','velin','velout','amx','amy','amz',\
                    'ekin','etot']):
                    self.rad[varname] = dat[0][rlocs]/norm
                elif (var in ['mdot','mdotin','mdotout',]):
                    self.rad[varname] = dat[0][rlocs]/self.rad['dr']
                else:
                    print(f"ERROR: No variable called {var}!!!")
        return
        
    def get_region(self,x0,x1,y0,y1,z0=0.0,z1=0.0):
        if (x0<x1):
            xloc=np.logical_and((self.coord['x']+0.5*self.coord['dx'])>x0,\
                                (self.coord['x']-0.5*self.coord['dx'])<x1,)
        else:
            xloc=np.logical_and((self.coord['x']+0.5*self.coord['dx'])>=x0,\
                                (self.coord['x']-0.5*self.coord['dx'])<=x1,)
        if (y0<y1):
            yloc=np.logical_and((self.coord['y']+0.5*self.coord['dy'])>y0,\
                                (self.coord['y']-0.5*self.coord['dy'])<y1,)
        else:
            yloc=np.logical_and((self.coord['y']+0.5*self.coord['dy'])>=y0,\
                                (self.coord['y']-0.5*self.coord['dy'])<=y1,)
        if (z0<z1):
            zloc=np.logical_and((self.coord['z']+0.5*self.coord['dz'])>z0,\
                                (self.coord['z']-0.5*self.coord['dz'])<z1,)
        else:
            zloc=np.logical_and((self.coord['z']+0.5*self.coord['dz'])>=z0,\
                                (self.coord['z']-0.5*self.coord['dz'])<=z1,)
        locs = np.where(np.logical_and(np.logical_and(xloc,yloc),zloc))
        return locs

    def plot_snapshot(self,var='dens',varname=None,weivar='vol',level=None,zoom=1,xyz=[],bins=None,\
                      title='',label='',xlabel='X',ylabel='Y',cmap='viridis',\
                      norm=LogNorm(1e-1,1e1),save=False,savepath='',dpi=200,**kwargs):
        fig=plt.figure(dpi=dpi)
        ax = plt.axes()
        bins=int(np.min([self.Nx1,self.Nx2,self.Nx3])) if not bins else bins
        if (not xyz):
            xyz = [self.x1min/zoom,self.x1max/zoom,self.x2min/zoom,self.x2max/zoom,0.0,0.0]
        x0,x1,y0,y1,z0,z1 = xyz[0],xyz[1],xyz[2],xyz[3],xyz[4],xyz[5]
        locs = self.get_region(x0,x1,y0,y1,z0,z1)
        if isinstance(weivar,str):
            weidata=self.coord[weivar][locs]
        else:
            weidata=weivar[locs]
        x=self.coord['x'][locs]
        y=self.coord['y'][locs]
        imgnorm = np.histogram2d(x,y,bins=bins,weights=weidata)[0]
        if isinstance(var,str):
            data=self.mb_data[var][locs]
            varname=var if not varname else varname
        else:
            data=var[locs]
        img = np.histogram2d(x,y,bins=bins,weights=weidata*data)[0]/imgnorm

        im=ax.imshow(img.swapaxes(0,1)[::-1,:],extent=(x0,x1,y0,y1),norm=norm,cmap=cmap,**kwargs)
        #im=ax.imshow(np.rot90(data),cmap='plasma',norm=LogNorm(0.9e-1,1.1e1),extent=extent)
        if (self.header('problem','r_in')): 
            ax.add_patch(plt.Circle((0,0),float(self.header('problem','r_in')),ec='k',fc='#00000000'))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Time = {self.time}" if not title else title)
        fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        if (save):
            fig.savefig(f"../figure/Simu_{self.path}/{self.label}/fig_{varname}_{self.num:04d}.png"\
                        if not savepath else savepath)
        return fig

    def set_dist(self,varl=['dens','temp','pres'],varsuf='',bins=100,weights='vol',\
        locs=None,update=False):
        for var in varl:
            varname = var+varsuf
            if (update or varname not in self.dist.keys()):
                weinorm = self.data(weights)
                if(locs is not None):
                    weinorm=weinorm*locs
        for var in varl:
            varname = var+varsuf
            if (update or varname not in self.dist.keys()):
                dat = np.histogram(np.log10(self.data(var)),bins=bins,weights=weinorm)
                self.dist[varname] = dat
        return 

    def set_dist2d(self,varl2d=[['dens','temp'],['dens','pres']],varsuf='',bins=100,\
        weights='vol',locs=None,update=False):
        weinorm = self.data(weights)
        if(locs is not None):
            weinorm=weinorm*locs
        for varl in varl2d:
            varname = varl[0]+"_"+varl[1]+varsuf
            if (update or varname not in self.dist2d.keys()):
                dats=[None,None]
                for i,var in enumerate(varl):
                    dats[i] = np.log10(self.data(var))
                self.dist2d[varname]=np.histogram2d(dats[0].ravel(),dats[1].ravel(),bins=bins,\
                    weights=weinorm.ravel())
        return 
