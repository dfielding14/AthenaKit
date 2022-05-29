import os
from pathlib import Path
import numpy as np
import struct
import h5py
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d

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
        if isinstance(item, (np.ndarray, np.int64, np.float64, np.bool_, str, int, float,
                             bytes, tuple, list)):
            h5file[path + str(key)] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            print('Error: ', key, ":", item)
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
    def __init__(self,binarypath,path='',label='',name='Base',nlim=10001,update=True,**kwargs):
        self.nlim=nlim
        self.alist=[]
        self.abins=[None]*self.nlim
        self.binarypath=binarypath
        self.path=path
        self.label=label
        self.name=name
        self.evo={}
        self.update=update
        return

    def read(self,ilist,info=False,redo=False):
        for i in ilist:
            if(redo or i not in self.alist):
                if(info): print("read:",i)
                if self.abins[i] is None:
                    self.abins[i]=AthenaBinary(path=self.path,label=self.label,num=i)
                self.abins[i].load_binary(self.binarypath+f"{i:05d}.bin")
            self.alist=sorted(list(set(self.alist + list([i]))))
        return

    def save_hdf5(self,ilist=None,hdf5path=None,info=False):
        ilist=self.alist if not ilist else ilist
        hdf5path=self.path+self.label+f"/data/hdf5/" if not hdf5path else hdf5path
        if not os.path.isdir(hdf5path):
            os.mkdir(hdf5path)
        for i in ilist:
            if(info): print("save hdf5:",i)
            abin=self.abins[i]
            abin.save_hdf5(hdf5path+f"{self.name}.{i:05d}.hdf5")
    
    def load_hdf5(self,ilist,hdf5path=None,info=False,redo=False):
        hdf5path=self.path+self.label+f"/data/hdf5/" if not hdf5path else hdf5path
        for i in ilist:
            if(redo or i not in self.alist):
                if(info): print("load hdf5:",i)
                if self.abins[i] is None:
                    self.abins[i]=AthenaBinary(path=self.path,label=self.label,num=i)
                self.abins[i].load_hdf5(hdf5path+f"{self.name}.{i:05d}.hdf5")
            self.alist=sorted(list(set(self.alist + list([i]))))
        return

    @property
    def abin(self):
        return self.abins[self.alist[0]]

    def config(self,redo=False,**kwargs):
        if(redo or not self.abin.config_flag):
            self.abin.config(**kwargs)
        for i in self.alist[1:]:
            # only set coord once
            abin=self.abins[i]
            if(redo or not abin.coord):
                abin.coord=self.abin.coord
            if(redo or not abin.config_flag):
                abin.config(**kwargs)
        return

    def set_hist(self,ilist=[],info=False,**kwargs):
        for i in (self.alist if not ilist else ilist):
            if(self.update or not self.abins[i].rad):
                if(info): print("set_hist:",i)
                self.abins[i].set_hist(**kwargs)
        return

    def map_hist(self,ilist=[]):
        self.hist={}
        self.hist['time']=np.array([self.abins[i].time for i in (self.alist if not ilist else ilist)])
        for var in self.abin.hist.keys():
            self.hist[var]=np.array([self.abins[i].hist[var] for i in (self.alist if not ilist else ilist)])
        return

    def set_radial(self,ilist=[],info=False,**kwargs):
        for i in (self.alist if not ilist else ilist):
            if(self.update or not self.abins[i].rad):
                if(info): print("set_radial:",i)
                self.abins[i].set_radial(**kwargs)
        return

    def set_slice(self,ilist=[],info=False,**kwargs):
        for i in (self.alist if not ilist else ilist):
            if(self.update or not self.abins[i].slice):
                if(info): print("set_slice:",i)
                self.abins[i].set_slice(**kwargs)
        return

    def set_dist(self,ilist=[],info=False,**kwargs):
        for i in (self.alist if not ilist else ilist):
            if(self.update or not self.abins[i].slice):
                if(info): print("set_dist:",i)
                self.abins[i].set_dist(**kwargs)
        return

    def set_dist2d(self,ilist=[],info=False,**kwargs):
        for i in (self.alist if not ilist else ilist):
            if(self.update or not self.abins[i].slice):
                if(info): print("set_dist2d:",i)
                self.abins[i].set_dist2d(**kwargs)
        return

    def set_spectrum(self,ilist=[],info=False,**kwargs):
        for i in (self.alist if not ilist else ilist):
            if(self.update or not self.abins[i].spectra):
                if(info): print("set_spectrum:",i)
                self.abins[i].set_spectrum(**kwargs)
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
        for i in self.alist:
            del self.abins[i]
        self.alist=[]
        self.abins=[None]*self.nlim
        return

    def set_evo(self,varl=['time','mdot'],nrin=1.0):
        for var in varl:
            if (var=='time'):
                self.evo[var]=np.array([self.abins[i].time for i in self.alist])
            elif (var=='mass'):
                self.evo[var]=np.array([np.sum((self.abins[i].rad['dens']*4.0*np.pi*\
                    self.abins[i].rad['r']**2*self.abins[i].rad['dr'])\
                    [np.logical_and(self.abins[i].rad['dens']>0,1)])\
                    for i in self.alist])
                self.evo[var]=np.array([self.abins[i].hist['dens'] for i in self.alist])
            elif (var=='m_warm'):
                self.evo[var]=np.array([np.sum((self.abins[i].rad['dens_warm']*4.0*np.pi*\
                    self.abins[i].rad['r']**2*self.abins[i].rad['dr'])\
                    [np.logical_and(self.abins[i].rad['dens_warm']>0,1)])\
                    for i in self.alist])
                self.evo[var]=np.array([self.abins[i].hist['dens_warm'] for i in self.alist])
            elif (var=='m_hot'):
                self.evo[var]=np.array([np.sum((self.abins[i].rad['dens_hot']*4.0*np.pi*\
                    self.abins[i].rad['r']**2*self.abins[i].rad['dr'])\
                    [np.logical_and(self.abins[i].rad['dens_hot']>0,1)])\
                    for i in self.alist])
                self.evo[var]=np.array([self.abins[i].hist['dens_hot'] for i in self.alist])
            else:
                irin = np.argmax(self.abin.rad['r']>nrin*float(self.abin.header('problem','r_in')))
                self.evo[var]=np.array([self.abins[i].rad[var][irin] for i in self.alist])

    def set_evo_dist(self,varl=['temp'],bins=128,density=None):
        self.evo_dist={}
        for var in varl:
            self.evo_dist[var]={}
            loc_l=np.min([self.abins[i].dist[var]['loc'][0] for i in self.alist])
            loc_u=np.max([self.abins[i].dist[var]['loc'][-1] for i in self.alist])
            newlocs = np.linspace(loc_l,loc_u,bins+1,endpoint=True)
            self.evo_dist[var]['dat']=np.empty((self.nlim,bins))
            self.evo_dist[var]['loc']=newlocs
            for i in self.alist:
                thisloc=self.abins[i].dist[var]['loc']
                locs=0.5*(thisloc[1:]+thisloc[:-1])
                fac=(newlocs[1]-newlocs[0])/(thisloc[1]-thisloc[0])
                dist=interp1d(locs,self.abins[i].dist[var]['dat'],bounds_error=False,fill_value=0.0)
                self.evo_dist[var]['dat'][i]=fac*dist(0.5*(newlocs[1:]+newlocs[:-1]))
                if (density == True):
                    self.evo_dist[var]['dat'][i]/=np.sum(self.evo_dist[var]['dat'][i])*(newlocs[1]-newlocs[0])
        return

    def avg_rad(self,ilist=None):
        ilist = self.alist if ilist is None else ilist
        self.rad={}
        for varname in self.abin.rad.keys():
            self.rad[varname]=np.mean([self.abins[i].rad[varname] for i in ilist],axis=0)
        return

    def plot_snapshot(self,ilist,figpath=None,info=False,figdir="../figure/Simu_",save=False,**kwargs):
        figpath=figdir+Path(self.path).parts[-1]+'/'+self.label+"/" if not figpath else figpath
        if save and not os.path.isdir(figpath):
            os.mkdir(figpath)
        for i in ilist:
            if (info): print("plot snapshot:",i)
            fig=self.abins[i].plot_snapshot(figdir=figdir,save=save,**kwargs)
            if (i!=ilist[-1]): plt.close(fig)
        return fig

    def plot_phase(self,ilist,figpath=None,info=False,figdir="../figure/Simu_",**kwargs):
        figpath=figdir+Path(self.path).parts[-1]+'/'+self.label+"/" if not figpath else figpath
        if not os.path.isdir(figpath):
            os.mkdir(figpath)
        for i in ilist:
            if (info): print("plot snapshot:",i)
            fig=self.abins[i].plot_phase(figdir=figdir,**kwargs)
        return fig

    def make_movie(self,varname,range,duration=0.05,fps=20,figdir="../figure/Simu_"):
        from moviepy.editor import ImageClip, concatenate_videoclips
        img = (f"{figdir}{Path(self.path).parts[-1]}/{self.label}/fig_{varname}_{i:04d}.png" for i in range)
        clips = [ImageClip(m).set_duration(duration) for m in img]
        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(f"{figdir}{Path(self.path).parts[-1]}/{self.label}/video_{varname}.mp4",fps=fps)
        return

class AthenaBinary:
    def __init__(self,path='',label='',num=0):
        self.path=path
        self.label=label
        self.num=num
        self._header={}
        self.raw={}
        self.coord={}
        self.user_data={}
        self.hist={}
        self.rad={}
        self.slice={}
        self.dist={}
        self.dist2d={}
        self.spectra={}
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
            if (k in ['raw', 'mb_data', 'coord', 'user_data']):
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
        #elif (var in ['dens','velx','vely','velz','velr','temp','eint']):
        elif (var in self.mb_data.keys()):
            return self.mb_data[var]
        elif (var in ['ones','mass','pres','entropy','momx','momy','momz','momtot','momr',\
            'velin','velout','vtot2','vtot','vrot','ekin','etot','amx','amy','amz','amtot',\
            'mdot','mdotin','mdotout']):
            if (var=='ones'):
                return np.ones(self.mb_data['dens'].shape)
            if (var=='mass'):
                return self.coord['vol']*self.mb_data['dens']
            elif (var=='pres'):
                return (self.gamma-1)*self.mb_data['eint']
            elif (var=='entropy'):
                return self.data('pres')/self.mb_data['dens']**self.gamma
            elif (var=='momx'):
                return self.mb_data['velx']*self.mb_data['dens']
            elif (var=='momy'):
                return self.mb_data['vely']*self.mb_data['dens']
            elif (var=='momz'):
                return self.mb_data['velz']*self.mb_data['dens']
            elif (var=='momr'):
                return self.mb_data['velr']*self.mb_data['dens']
            elif (var=='velin'):
                return np.minimum(self.mb_data['velr'],0.0)
            elif (var=='velout'):
                return np.maximum(self.mb_data['velr'],0.0)
            elif (var=='vtot2'):
                return self.data('velx')**2+self.data('vely')**2+self.data('velz')**2
            elif (var=='vtot'):
                return np.sqrt(self.data('vtot2'))
            elif (var=='vrot'):
                return np.sqrt(self.data('vtot2')-self.data('velr')**2)
            elif (var=='momtot'):
                return self.mb_data['dens']*self.data('vtot')
            elif (var=='ekin'):
                return 0.5*self.mb_data['dens']*self.data('vtot2')
            elif (var=='etot'):
                return self.data('ekin')+self.mb_data['eint']
            elif (var=='amx'):
                return self.data('y')*self.data('velz')-self.data('z')*self.data('vely')
            elif (var=='amy'):
                return self.data('z')*self.data('velx')-self.data('x')*self.data('velz')
            elif (var=='amz'):
                return self.data('x')*self.data('vely')-self.data('y')*self.data('velx')
            elif (var=='amtot'):
                return self.data('r')*self.data('vrot')
            elif (var=='mdot'):
                return self.mb_data['dens']*self.mb_data['velr']
            elif (var=='mdotin'):
                return self.mb_data['dens']*self.data('velin')
            elif (var=='mdotout'):
                return self.mb_data['dens']*self.data('velout')
            else:
                print(f"ERROR: No data callled {var}!!!")
        # user vars
        elif (var in self.user_data.keys()):
            return self.user_data[var]
        else:
            print(f"ERROR: No data callled {var}!!!")
        return None

    def get_coord(self,level=0,xyz=[]):
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
        # TODO(@mhguo)
        print(nx1_fac,i_min,i_max)
        print(nx2_fac,j_min,j_max)
        print(nx3_fac,k_min,k_max)
        data = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min))
        x=np.linspace(xyz[0],xyz[1],i_max-i_min)
        y=np.linspace(xyz[2],xyz[3],j_max-j_min)
        z=np.linspace(xyz[4],xyz[5],k_max-k_min)
        dx=(xyz[1]-xyz[0])/(i_max-i_min)
        dy=(xyz[3]-xyz[2])/(j_max-j_min)
        dz=(xyz[5]-xyz[4])/(k_max-k_min)
        ZYX=np.meshgrid(z,y,x)
        return ZYX[2].swapaxes(0,1),ZYX[1].swapaxes(0,1),ZYX[0].swapaxes(0,1),dx,dy,dz
        self.coord['r']=np.sqrt(self.coord['x']**2+self.coord['y']**2+self.coord['z']**2)
        dx=np.asarray([np.full((self.nx3_mb,self.nx2_mb,self.nx1_mb),self.mb_geometry[i,3]) for i in range(self.n_mbs)])
        dy=np.asarray([np.full((self.nx3_mb,self.nx2_mb,self.nx1_mb),self.mb_geometry[i,4]) for i in range(self.n_mbs)])
        dz=np.asarray([np.full((self.nx3_mb,self.nx2_mb,self.nx1_mb),self.mb_geometry[i,5]) for i in range(self.n_mbs)])
        self.coord['dx'],self.coord['dy'],self.coord['dz']=dx,dy,dz
        self.coord['vol']=self.coord['dx']*self.coord['dy']*self.coord['dz']
        return

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

    def set_hist(self,varl=['dens','temp','velr','mdot'],varsuf='',\
        locs=None,weights='vol',norm=1.0,redo=False):
        for var in varl:
            varname = var+varsuf
            if (redo or varname not in self.hist.keys()):
                if (locs is None):
                    dat = np.sum(self.data(var)*self.data(weights))
                else:
                    dat = np.sum((self.data(var)*self.data(weights))[locs])
                self.hist[varname] = dat*norm
        return

    def set_radial(self,varl=['dens','temp','velr','mdot'],varsuf='',bins=1000,\
        locs=None,weights='vol',redo=False):
        for var in varl:
            varname = var+varsuf
            if (redo or varname not in self.rad.keys()):
                logr = np.log10(self.coord['r'])
                weinorm = self.data(weights)
                hist = np.histogram(logr,bins=bins,weights=weinorm)
                self.rad['r'] = (10**((hist[1][:-1]+hist[1][1:])/2))
                r_range = (hist[1][0],hist[1][-1])
                r_locs = np.logical_and(hist[0]!=0,self.rad['r']<self.rmax)
                self.rad['r'] = self.rad['r'][r_locs]
                dr_locs = np.append(r_locs,True)
                self.rad['dr'] = (10**((hist[1][dr_locs])[1:])-10**((hist[1][dr_locs])[:-1]))
                if (locs is None):
                    hist2 = hist
                else:
                    hist2 = np.histogram(logr[locs],range=r_range,bins=bins,weights=weinorm[locs])
                norm = hist2[0][r_locs]
                self.rad['norm'+varsuf]=norm
                break
        for var in varl:
            varname = var+varsuf
            if (redo or varname not in self.rad.keys()):
                if (locs is None):
                    dat = np.histogram(logr,range=r_range,bins=bins,weights=weinorm*self.data(var))
                else:
                    dat = np.histogram(logr[locs],range=r_range,bins=bins,weights=weinorm[locs]*self.data(var)[locs])
                if (var in ['mdot','mdotin','mdotout',]):
                    self.rad[varname] = dat[0][r_locs]/self.rad['dr']
                else:
                    self.rad[varname] = dat[0][r_locs]/norm
        return

    def get_slice_coord(self,zoom=0,level=0,xyz=[],axis=0):
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**level/self.Nx3,self.x3max/2**level/self.Nx3]
        x,y,z,dx,dy,dz=self.get_coord(level=level,xyz=xyz)
        return np.average(x,axis=axis),np.average(y,axis=axis),np.average(z,axis=axis),xyz

    def get_slice(self,var='dens',zoom=0,level=0,xyz=[],axis=0):
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**level/self.Nx3,self.x3max/2**level/self.Nx3]
        return np.average(self.get_data(var,level=level,xyz=xyz),axis=axis),xyz

    def set_slice(self,varl=['dens','temp'],varsuf='',zoom=0,level=0,xyz=[],redo=False):
        for var in varl:
            varname = var+varsuf
            if (redo or varname not in self.slice.keys()):
                self.slice[varname] = {}
                data = self.get_slice(var,zoom=zoom,level=level,xyz=xyz)
                self.slice[varname]['dat'] = data[0]
                self.slice[varname]['xyz'] = data[1]

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

    def set_dist(self,varl=['dens','temp','pres'],varsuf='',scale='log',bins=128,weights='vol',\
        locs=None,redo=False):
        for var in varl:
            varname = var+varsuf
            if (redo or varname not in self.dist.keys()):
                weinorm = self.data(weights) if type(weights) is str else weights
        for var in varl:
            varname = var+varsuf
            if (redo or varname not in self.dist.keys()):
                if (scale=='log'):
                    data = np.log10(self.data(var))
                else:
                    data = self.data(var)
                if (locs is None):
                    dat = np.histogram(data,bins=bins,weights=weinorm)
                else:
                    dat = np.histogram(data[locs],bins=bins,weights=weinorm[locs])
                self.dist[varname] = {}
                self.dist[varname]['dat'] = dat[0]
                self.dist[varname]['loc'] = dat[1]
        return 

    def set_dist2d(self,varl2d=[['dens','temp'],['dens','pres']],scales=['log','log'],\
        varsuf='',bins=128,weights='vol',locs=None,redo=False):
        for varl in varl2d:
            varname = varl[0]+"_"+varl[1]+varsuf
            if (redo or varname not in self.dist2d.keys()):
                weinorm = self.data(weights) if type(weights) is str else weights
                dats=[None,None]
                for i,var in enumerate(varl):
                    if (scales[i]=='log'):
                        dats[i] = np.log10(self.data(var))
                        dats[i][dats[i]==-np.inf]=0.0
                    else:
                        dats[i] = self.data(var)
                if (locs is None):
                    dat = np.histogram2d(dats[0].ravel(),dats[1].ravel(),bins=bins,\
                        weights=weinorm.ravel())
                else:
                    dat = np.histogram2d(dats[0][locs],dats[1][locs],bins=bins,\
                        weights=weinorm[locs])
                self.dist2d[varname] = {}
                self.dist2d[varname]['dat'] = dat[0]
                self.dist2d[varname]['loc1'] = dat[1]
                self.dist2d[varname]['loc2'] = dat[2]
        return 

    def plot_snapshot(self,var='dens',varname='',zoom=0,level=0,xyz=[],unit=1.0,bins=None,\
                      title='',label='',xlabel='X',ylabel='Y',cmap='viridis',\
                      norm=LogNorm(1e-1,1e1),save=False,figdir='../figure/Simu_',\
                      savepath='',savelabel='',figlabel='',dpi=200,vel=None,stream=None,circle=True,\
                      fig=None,ax=None,xyunit=1.0,**kwargs):
        fig=plt.figure(dpi=dpi) if fig is None else fig
        ax = plt.axes() if ax is None else ax
        bins=int(np.min([self.Nx1,self.Nx2,self.Nx3])) if not bins else bins
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**level/self.Nx3*2,self.x3max/2**level/self.Nx3*2]
        if varname in self.slice.keys():
            slc = self.slice[varname]['dat']*unit
            xyz = self.slice[varname]['xyz']
        else:
            slc = self.get_slice(var,zoom=zoom,level=level,xyz=xyz)[0]*unit
        x0,x1,y0,y1,z0,z1 = xyz[0],xyz[1],xyz[2],xyz[3],xyz[4],xyz[5]
        
        im=ax.imshow(slc[::-1,:],extent=(x0*xyunit,x1*xyunit,y0*xyunit,y1*xyunit),\
            norm=norm,cmap=cmap,**kwargs)
        if (vel is not None):
            x,y,z = self.get_slice_coord(zoom=zoom,level=vel,xyz=xyz)[:3]
            #x = self.get_slice('x',zoom=zoom,level=vel,xyz=xyz)[0]
            #y = self.get_slice('y',zoom=zoom,level=vel,xyz=xyz)[0]
            u = self.get_slice('velx',zoom=zoom,level=vel,xyz=xyz)[0]
            v = self.get_slice('vely',zoom=zoom,level=vel,xyz=xyz)[0]
            ax.quiver(x*xyunit, y*xyunit, u, v)
        if (stream is not None):
            x,y,z = self.get_slice_coord(zoom=zoom,level=stream,xyz=xyz)[:3]
            #x = self.get_slice('x',zoom=zoom,level=stream,xyz=xyz)[0]
            #y = self.get_slice('y',zoom=zoom,level=stream,xyz=xyz)[0]
            u = self.get_slice('velx',zoom=zoom,level=stream,xyz=xyz)[0]
            v = self.get_slice('vely',zoom=zoom,level=stream,xyz=xyz)[0]
            #x,y=np.meshgrid(x,y)
            #z,x=np.meshgrid(z,x)
            #np.mgrid[-w:w:100j, -w:w:100j]
            #beg=8
            #step=16
            #ax.streamplot(x[beg::step,beg::step], z[beg::step,beg::step], (u[0]/norm[0])[beg::step,beg::step], (v[0]/norm[0])[beg::step,beg::step])
            ax.streamplot(x*xyunit, y*xyunit, u, v,color='k')

        #im=ax.imshow(slc.swapaxes(0,1)[::-1,:],extent=(x0,x1,y0,y1),norm=norm,cmap=cmap,**kwargs)
        #im=ax.imshow(np.rot90(data),cmap='plasma',norm=LogNorm(0.9e-1,1.1e1),extent=extent)
        if(circle and self.header('problem','r_in')):
            ax.add_patch(plt.Circle((0,0),float(self.header('problem','r_in')),ec='k',fc='#00000000'))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Time = {self.time}" if not title else title)
        fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        if (save):
            fig.savefig(f"{figdir}{Path(self.path).parts[-1]}/{self.label}/fig_{varname+figlabel if not savelabel else savelabel}_{self.num:04d}.png"\
                        if not savepath else savepath, bbox_inches='tight')
        return fig

    def plot_phase(self,varname='dens_temp',title='',label='',xlabel='X',ylabel='Y',cmap='viridis',\
                   norm=LogNorm(1e-3,1e1),extent=None,save=False,savepath='',figdir='../figure/Simu_',\
                   fig=None,ax=None,dpi=128,aspect='auto',**kwargs):
        fig=plt.figure(dpi=dpi) if fig is None else fig
        ax = plt.axes() if ax is None else ax
        dat = self.dist2d[varname]
        extent = [dat['loc1'].min(),dat['loc1'].max(),dat['loc2'].min(),dat['loc2'].max()] if extent is None else extent
        im = ax.imshow(dat['dat'].swapaxes(0,1)[::-1,:],extent=extent,norm=norm,cmap=cmap,aspect=aspect,**kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Time = {self.time}" if not title else title)
        fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        return fig

    # TODO (@mhguo): modify this!
    def plot_snapshots(self,var='dens',varname='',zoom=0,level=0,xyz=[],unit=1.0,bins=None,\
                      title='',label='',xlabel='X',ylabel='Y',cmap='viridis',\
                      norm=LogNorm(1e-1,1e1),save=False,figdir='../figure/Simu_',\
                      savepath='',figlabel='',dpi=128,vel=None,stream=None,circle=True,\
                      fig=None,xyunit=1.0,nrows=2,ncols=2,figsize=(8,8),\
                      sharex=True,squeeze=False,constrained_layout=False,\
                      top=0.94, wspace=0.5, hspace=0.0,**kwargs):
        fig,axes=plt.subplots(nrows,ncols,figsize=figsize,dpi=dpi,sharex=sharex,\
                constrained_layout=constrained_layout,squeeze=squeeze) if fig is None else fig
        fig.subplots_adjust(top=top, wspace=wspace, hspace=hspace)
        #fig.suptitle(f'cycle={ab.abins[i].cycle}')
        self.plot_snapshot(fig=fig,ax=axes[0,0],var='dens',label=r'$n\rm\,[cm^{-3}]$',\
            cmap='viridis',xlabel='X [pc]',ylabel='Y [pc]',vel=vel,level=level,varname='dtmp',\
            xyz=xyz,circle=False,unit=1,xyunit=1e3,dpi=dpi,norm=LogNorm(1e-3,5e3))
        self.plot_snapshot(fig=fig,ax=axes[0,1],var='temp',label=r'$T\rm\,[K]$',\
            cmap='hot',    xlabel='X [pc]',ylabel='',       vel=vel,level=level,varname='dtmp',\
            xyz=xyz,circle=False,unit=1,xyunit=1e3,dpi=dpi,norm=LogNorm(1e5,1e9))
        self.plot_snapshot(fig=fig,ax=axes[1,0],var='momtot',label=r'$|p|\rm\,[10^3\,km\,s^{-1}\,cm^{-3}]$',\
            cmap='cividis',xlabel='X [pc]',ylabel='Y [pc]',vel=vel,level=level,varname='dtmp',\
            xyz=xyz,circle=False,unit=1,xyunit=1e3,dpi=dpi,norm=LogNorm(1e-3,5e3))
        self.plot_snapshot(fig=fig,ax=axes[1,1],var='pres',label=r'$P\rm\,[k_B\,K\,cm^{-3}]$',\
            cmap='plasma',xlabel='X [pc]',ylabel='',       vel=vel,level=level,varname='dtmp',\
            xyz=xyz,circle=False,unit=1,xyunit=1e3,dpi=dpi,norm=LogNorm(1e5,1e9))
        return

    def get_data_new_coord(self,var='dens',orient=[0,0,1],zoom=0,level=0,xyz=[]):
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**zoom,self.x3max/2**zoom,]
        orient=orient/np.sqrt(np.sum(np.array(orient)**2))
        x,y,z,dx,dy,dz=self.get_coord(level=level,xyz=xyz)
        r=np.sqrt(x**2+y**2+z**2)
        oz=x*orient[0]+y*orient[1]+z*orient[2]
        ox=np.sqrt(r**2-oz**2)
        if (var=='v_phi'):
            dat=(orient[0]*self.get_data('amx',level=level,xyz=xyz)+\
                 orient[1]*self.get_data('amy',level=level,xyz=xyz)+\
                 orient[2]*self.get_data('amz',level=level,xyz=xyz))/ox
        elif (var=='v_oz'):
            dat=orient[0]*self.get_data('velx',level=level,xyz=xyz)+\
                orient[1]*self.get_data('vely',level=level,xyz=xyz)+\
                orient[2]*self.get_data('velz',level=level,xyz=xyz)
        elif (var=='v_ox'):
            xvx=x*self.get_data('velx',level=level,xyz=xyz)
            yvy=y*self.get_data('vely',level=level,xyz=xyz)
            zvz=z*self.get_data('velz',level=level,xyz=xyz)
            #dat=(orient[0]*(yvy+zvz)+orient[1]*(zvz+xvx)+orient[2]*(xvx+yvy))/ox
            dat=(xvx*(orient[1]+orient[2])+yvy*(orient[2]+orient[0])+zvz*(orient[0]+orient[1]))/ox
        else:
            dat=self.get_data(var,level=level,xyz=xyz)
        return dat

    def get_phi_averaged(self,var='dens',orient=[0,0,1],zoom=0,level=0,bins=256,xyz=[],xz=[]):
        print("get_phi_averaged: "+var)
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**zoom,self.x3max/2**zoom,]
        if (not xz):
            xz = [self.x1max/2**zoom,self.x3max/2**zoom]
        orient=orient/np.sqrt(np.sum(np.array(orient)**2))
        x,y,z,dx,dy,dz=self.get_coord(level=level,xyz=xyz)
        dvol=dx*dy*dz
        r=np.sqrt(x**2+y**2+z**2)
        oz=x*orient[0]+y*orient[1]+z*orient[2]
        ox=np.sqrt(r**2-oz**2)
        locs=np.logical_and(ox<xz[0],(oz-xz[1])*(oz+xz[1])<0)
        norm = np.histogram2d(ox[locs],oz[locs],bins=bins)
        if (var=='v_phi'):
            dat=(orient[0]*self.get_data('amx',level=level,xyz=xyz)+\
                 orient[1]*self.get_data('amy',level=level,xyz=xyz)+\
                 orient[2]*self.get_data('amz',level=level,xyz=xyz))/ox
        elif (var=='v_oz'):
            dat=orient[0]*self.get_data('velx',level=level,xyz=xyz)+\
                orient[1]*self.get_data('vely',level=level,xyz=xyz)+\
                orient[2]*self.get_data('velz',level=level,xyz=xyz)
        elif (var=='v_ox'):
            xvx=x*self.get_data('velx',level=level,xyz=xyz)
            yvy=y*self.get_data('vely',level=level,xyz=xyz)
            zvz=z*self.get_data('velz',level=level,xyz=xyz)
            #dat=(orient[0]*(yvy+zvz)+orient[1]*(zvz+xvx)+orient[2]*(xvx+yvy))/ox
            dat=(xvx*(orient[1]+orient[2])+yvy*(orient[2]+orient[0])+zvz*(orient[0]+orient[1]))/ox
        else:
            dat=self.get_data(var,level=level,xyz=xyz)
        img = np.histogram2d(ox[locs],oz[locs],bins=bins,weights=dat[locs])
        return img,norm

    def fft_comp(self, data, nindex=1):
        nz, ny, nx, = data.shape

        # do the FFTs -- note that since our data is real, there will be
        # too much information here.  fftn puts the positive freq terms in
        # the first half of the axes -- that's what we keep.  Our
        # normalization has an '8' to account for this clipping to one
        # octant.
        ru = np.fft.fftn(data ** nindex)[
            0 : nz // 2 + 1, 0 : ny // 2 + 1, 0 : nx // 2 + 1
        ]
        ru = 8.0 * ru / (nx * ny * nz)

        return np.abs(ru) ** 2

    def get_spectrum(self, var):
        nz, ny, nx = dims = np.asarray([self.Nx3,self.Nx2,self.Nx1])
        L = np.array([(self.x1max-self.x1min),(self.x2max-self.x2min),(self.x3max-self.x3min)])
        kx = np.fft.rfftfreq(nx) * nx / L[0]
        ky = np.fft.rfftfreq(ny) * ny / L[1]
        kz = np.fft.rfftfreq(nz) * nz / L[2]
        kmin = np.min(1.0 / L)
        kmax = np.min(0.5 * dims / L)
        kbins = np.arange(kmin, kmax+kmin, kmin)
        N = len(kbins)
        kz3d, ky3d, kx3d = np.meshgrid(kz, ky, kx, indexing="ij")
        k = np.sqrt(kx3d ** 2 + ky3d ** 2 + kz3d ** 2)

        Kk=self.fft_comp(self.get_data(var))
        
        '''
        whichbin = np.digitize(k.flat, kbins)
        ncount = np.bincount(whichbin)
        E_spectrum = np.zeros(len(ncount) - 1)
        for n in range(1, len(ncount)):
            E_spectrum[n - 1] = np.sum(Kk.flat[whichbin == n])
        #'''

        E_spectrum = np.histogram(k,bins=kbins,weights=Kk)[0]

        k = 0.5 * (kbins[0 : N - 1] + kbins[1:N])[:-1]
        E_spectrum = E_spectrum[1:N]
        return k, E_spectrum

    def set_spectrum(self,varl,**kwargs):
        for var in varl:
            k, spectrum = self.get_spectrum(var=var,**kwargs)
            self.spectra[var] = {}
            self.spectra[var]['k']=k
            self.spectra[var]['spectrum']=spectrum
        return
