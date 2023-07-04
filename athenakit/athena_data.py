import os
from pathlib import Path
import numpy as np
import h5py
import warnings
from packaging.version import parse as version_parse

from matplotlib import pyplot as plt

from .io import read_binary
from .utils import save_dict_to_hdf5, load_dict_from_hdf5

def load(filename):
    ad = AthenaData(filename)
    ad.load(filename,config=True)
    return ad

class AthenaData:
    def __init__(self,num=0,version='1.0'):
        self.num=num
        self.version=version
        self._header={}
        self.binary={}
        self.coord={}
        self.mb_data={}
        self.data_func={}
        self.hist={} # TODO(@mhguo): change to sum?
        self.rad={}
        self.slice={}
        self.dist={}
        self.dist2d={}
        self.spectra={}
        return
    
    # TODO(@mhguo): write a correct function to load data
    def load(self,filename,config=True):
        self.filename=filename
        if (filename.endswith('.bin')):
            self.binary_name = filename
            self.load_binary(filename)
        elif (filename.endswith('.athdf')):
            self.athdf_name = filename
            self.load_athdf(filename)
        elif (filename.endswith('.hdf5')):
            self.hdf5_name = filename
            self.load_hdf5(filename)
        if (config):
            self.config()
        return

    def load_binary(self,filename):
        self._load_from_binary(read_binary(filename))
        return

    # TODO(@mhguo): write a correct function to load data from athena++ hdf5 data
    def load_athdf(self,filename):
        self._load_from_athdf(filename)
        return

    def load_hdf5(self,filename):
        self._load_from_dic(load_dict_from_hdf5(filename))
        return
    
    def save_hdf5(self,filename):
        dic={}
        for k,v in self.__dict__.items():
            if (k in ['binary', 'coord', 'mb_data', 'h5file', 'h5dic']):
                continue
            dic[k]=v
        save_dict_to_hdf5(dic,filename)
        return

    def _load_from_dic(self,dic,except_keys=['header', 'data', 'binary', 'coord', 'mb_data']):
        for k,v in dic.items():
            if (k not in except_keys):
                self.__dict__[k]=v
        return

    def _load_from_binary(self,binary):
        self.binary = binary
        self._load_from_dic(self.binary)
        for var in self.binary['var_names']:
            self.mb_data[var]=np.asarray(binary['mb_data'][var])
        self._config_header(self.binary['header'])
        self._config_attrs_from_header()
        return
    
    def _load_from_athdf(self,filename):
        h5file = h5py.File(filename, mode='r')
        self.h5file = h5file
        self.h5dic = load_dict_from_hdf5(filename)
        self._config_header(self.h5file.attrs['Header'])
        self._config_attrs_from_header()
        self.time = self.h5file.attrs['Time']
        self.cycle = self.h5file.attrs['NumCycles']
        self.n_mbs = self.h5file.attrs['NumMeshBlocks']
        self.mb_logical = np.append(self.h5dic['LogicalLocations'],self.h5dic['Levels'].reshape(-1,1),axis=1)
        self.mb_geometry = np.asarray([self.h5dic['x1f'][:,0],self.h5dic['x1f'][:,-1],
                                       self.h5dic['x2f'][:,0],self.h5dic['x2f'][:,-1],
                                       self.h5dic['x3f'][:,0],self.h5dic['x3f'][:,-1],]).T
        n_var_read = 0
        for ds_n,num in enumerate(self.h5file.attrs['NumVariables']):
            for i in range(num):
                var = self.h5file.attrs['VariableNames'][n_var_read+i].decode("utf-8")
                self.mb_data[var] = self.h5dic[self.h5file.attrs['DatasetNames'][ds_n].decode("utf-8")][i]
            n_var_read += num
        return

    def config(self):
        if (not self.coord): self.set_coord()
        self._config_data_func()
        self.path = Path(self.filename).parent
        self.num = int(self.filename.split('.')[-2])
        # assuming use_e=True
        # assuming we have dens, velx, vely, velz, eint
        # TODO(@mhguo): add support for arbitrary variables
        return

    def _config_header(self, header):
        for line in [entry for entry in header]:
            if line.startswith('<'):
                block = line
                self._header[block]={}
                continue
            key, value = line.split('=')
            self._header[block][key.strip()] = value

    def header(self, blockname, keyname, astype=str, default=None):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if not blockname.startswith('<'):
            blockname = '<' + blockname
        if blockname[-1] != '>':
            blockname += '>'
        if blockname in self._header.keys():
            if keyname in self._header[blockname].keys():
                return astype(self._header[blockname][keyname])
        warnings.warn(f'Warning: no parameter called {blockname}/{keyname}, return default value: {default}')
        return default
    
    def _config_attrs_from_header(self):
        self.Nx1 = self.header( 'mesh', 'nx1', int)
        self.Nx2 = self.header( 'mesh', 'nx2', int)
        self.Nx3 = self.header( 'mesh', 'nx3', int)
        self.nx1 = self.header( 'meshblock', 'nx1', int)
        self.nx2 = self.header( 'meshblock', 'nx2', int)
        self.nx3 = self.header( 'meshblock', 'nx3', int)
        self.nghost = self.header( 'mesh', 'nghost', int)
        self.x1min = self.header( 'mesh', 'x1min', float)
        self.x1max = self.header( 'mesh', 'x1max', float)
        self.x2min = self.header( 'mesh', 'x2min', float)
        self.x2max = self.header( 'mesh', 'x2max', float)
        self.x3min = self.header( 'mesh', 'x3min', float)
        self.x3max = self.header( 'mesh', 'x3max', float)

        self.use_e=self.header('hydro','use_e',bool,True) if 'hydro' in self._header.keys() else self.header('mhd','use_e',bool,True) 
        self.gamma=self.header('hydro','gamma',float,5/3) if 'hydro' in self._header.keys() else self.header('mhd','gamma',float,5/3)
        
        return
    
    def set_coord(self):
        mb_geo, n_mbs = self.mb_geometry, self.n_mbs
        nx1, nx2, nx3 = self.nx1, self.nx2, self.nx3
        x=np.swapaxes(np.linspace(mb_geo[:,0],mb_geo[:,1],nx1+1),0,1)
        y=np.swapaxes(np.linspace(mb_geo[:,2],mb_geo[:,3],nx2+1),0,1)
        z=np.swapaxes(np.linspace(mb_geo[:,4],mb_geo[:,5],nx3+1),0,1)
        x,y,z=0.5*(x[:,:-1]+x[:,1:]),0.5*(y[:,:-1]+y[:,1:]),0.5*(z[:,:-1]+z[:,1:])
        ZYX=np.swapaxes(np.asarray([np.meshgrid(z[i],y[i],x[i]) for i in range(n_mbs)]),0,1)
        self.coord['x'],self.coord['y'],self.coord['z']=ZYX[2].swapaxes(1,2),ZYX[1].swapaxes(1,2),ZYX[0].swapaxes(1,2)
        dx=np.asarray([np.full((nx3,nx2,nx1),(mb_geo[i,1]-mb_geo[i,0])/nx1) for i in range(n_mbs)])
        dy=np.asarray([np.full((nx3,nx2,nx1),(mb_geo[i,3]-mb_geo[i,2])/nx2) for i in range(n_mbs)])
        dz=np.asarray([np.full((nx3,nx2,nx1),(mb_geo[i,5]-mb_geo[i,4])/nx3) for i in range(n_mbs)])
        self.coord['dx'],self.coord['dy'],self.coord['dz']=dx,dy,dz
        return

    ### data handling ###
    def add_data(self,name,func):
        self.data_func[name]=func
        return
    
    def _config_data_func(self):
        self.data_func['ones'] = lambda self : np.ones(self.data('dens').shape)
        self.data_func['vol'] = lambda self : self.data('dx')*self.data('dy')*self.data('dz')
        self.data_func['r'] = lambda self : np.sqrt(self.data('x')**2+self.data('y')**2+self.data('z')**2)
        self.data_func['mass'] = lambda self : self.coord['vol']*self.data('dens')
        self.data_func['pres'] = lambda self : (self.gamma-1)*self.data('eint')
        self.data_func['temp'] = lambda self : (self.gamma-1)*self.data('eint')/self.data('dens')
        self.data_func['entropy'] = lambda self : self.data('pres')/self.data('dens')**self.gamma
        self.data_func['momx'] = lambda self : self.data('velx')*self.data('dens')
        self.data_func['momy'] = lambda self : self.data('vely')*self.data('dens')
        self.data_func['momz'] = lambda self : self.data('velz')*self.data('dens')
        self.data_func['velr'] = lambda self : (self.data('velx')*self.data('x')+\
                                               self.data('vely')*self.data('y')+\
                                               self.data('velz')*self.data('z'))/self.data('r')
        self.data_func['momr'] = lambda self : self.data('velr')*self.data('dens')
        self.data_func['velin'] = lambda self : np.minimum(self.data('velr'),0.0)
        self.data_func['velout'] = lambda self : np.maximum(self.data('velr'),0.0)
        self.data_func['vtot2'] = lambda self : self.data('velx')**2+self.data('vely')**2+self.data('velz')**2
        self.data_func['vtot'] = lambda self : np.sqrt(self.data('vtot2'))
        self.data_func['vrot'] = lambda self : np.sqrt(self.data('vtot2')-self.data('velr')**2)
        self.data_func['momtot'] = lambda self : self.data('dens')*self.data('vtot')
        self.data_func['ekin'] = lambda self : 0.5*self.data('dens')*self.data('vtot2')
        self.data_func['etot'] = lambda self : self.data('ekin')+self.data('eint')
        self.data_func['amx'] = lambda self : self.data('y')*self.data('velz')-self.data('z')*self.data('vely')
        self.data_func['amy'] = lambda self : self.data('z')*self.data('velx')-self.data('x')*self.data('velz')
        self.data_func['amz'] = lambda self : self.data('x')*self.data('vely')-self.data('y')*self.data('velx')
        self.data_func['amtot'] = lambda self : self.data('r')*self.data('vrot')
        self.data_func['mdot'] = lambda self : self.data('dens')*self.data('velr')
        self.data_func['mdotin'] = lambda self : self.data('dens')*self.data('velin')
        self.data_func['mdotout'] = lambda self : self.data('dens')*self.data('velout')
        self.data_func['momflx'] = lambda self : self.data('dens')*self.data('velr')*self.data('velr')
        self.data_func['momflxin'] = lambda self : self.data('dens')*self.data('velr')*self.data('velin')
        self.data_func['momflxout'] = lambda self : self.data('dens')*self.data('velr')*self.data('velout')
        self.data_func['ekflx'] = lambda self : self.data('dens')*.5*self.data('vtot2')*self.data('velr')
        self.data_func['ekflxin'] = lambda self : self.data('dens')*.5*self.data('vtot2')*self.data('velin')
        self.data_func['ekflxout'] = lambda self : self.data('dens')*.5*self.data('vtot2')*self.data('velout')
        self.data_func['bccr'] = lambda self : (self.data('bcc1')*self.data('x')+\
                                                self.data('bcc2')*self.data('y')+\
                                                self.data('bcc3')*self.data('z'))/self.data('r')
        self.data_func['btot2'] = lambda self : self.data('bcc1')**2+self.data('bcc2')**2+self.data('bcc3')**2
        self.data_func['btot'] = lambda self : np.sqrt(self.data('btot2'))
        self.data_func['brot'] = lambda self : np.sqrt(self.data('btot2')-self.data('bccr')**2)
        return

    def data(self,var):
        if (var in self.coord.keys()):
            return self.coord[var]
        elif (var in self.mb_data.keys()):
            return self.mb_data[var]
        # derived vars
        elif (var in self.data_func.keys()):
            return self.data_func[var](self)
        else:
            raise ValueError(f"No variable callled '{var}' ")

    ### uniform data ###
    # TODO(@mhguo): deal with the cell center and cell edge issue
    def uni_coord(self,level=0,xyz=[]):
        if (not xyz):
            xyz = [self.x1min,self.x1max,self.x2min,self.x2max,self.x3min,self.x3max]
        # level is physical level
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
        #print("coord: ",nx1_fac,i_min,i_max)
        #print("coord: ",nx2_fac,j_min,j_max)
        #print("coord: ",nx3_fac,k_min,k_max)
        #data = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min))
        x=np.linspace(xyz[0],xyz[1],i_max-i_min)
        y=np.linspace(xyz[2],xyz[3],j_max-j_min)
        z=np.linspace(xyz[4],xyz[5],k_max-k_min)
        dx=(xyz[1]-xyz[0])/(i_max-i_min)
        dy=(xyz[3]-xyz[2])/(j_max-j_min)
        dz=(xyz[5]-xyz[4])/(k_max-k_min)
        ZYX=np.meshgrid(z,y,x)
        return ZYX[2].swapaxes(0,1),ZYX[1].swapaxes(0,1),ZYX[0].swapaxes(0,1),dx,dy,dz

    def uni_data(self,var,level=0,xyz=[]):
        if (not xyz):
            xyz = [self.x1min,self.x1max,self.x2min,self.x2max,self.x3min,self.x3max]
        # block_level is physical level of mesh refinement
        physical_level = level
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
            if (block_level <= physical_level):
                s = int(2**(physical_level - block_level))
                # Calculate destination indices, without selection
                il_d = block_loc[0] * self.nx1 * s if self.Nx1 > 1 else 0
                jl_d = block_loc[1] * self.nx2 * s if self.Nx2 > 1 else 0
                kl_d = block_loc[2] * self.nx3 * s if self.Nx3 > 1 else 0
                iu_d = il_d + self.nx1 * s if self.Nx1 > 1 else 1
                ju_d = jl_d + self.nx2 * s if self.Nx2 > 1 else 1
                ku_d = kl_d + self.nx3 * s if self.Nx3 > 1 else 1
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
                s = int(2 ** (block_level - physical_level))
                # Calculate destination indices, without selection
                il_d = int(block_loc[0] * self.nx1 / s) if self.Nx1 > 1 else 0
                jl_d = int(block_loc[1] * self.nx2 / s) if self.Nx2 > 1 else 0
                kl_d = int(block_loc[2] * self.nx3 / s) if self.Nx3 > 1 else 0
                iu_d = int(il_d + self.nx1 / s) if self.Nx1 > 1 else 1
                ju_d = int(jl_d + self.nx2 / s) if self.Nx2 > 1 else 1
                ku_d = int(kl_d + self.nx3 / s) if self.Nx3 > 1 else 1
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
    
    ### histogram ###

    ### radial profile ###
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
                if (var in ['mdot','mdotin','mdotout','momflx','momflxin','momflxout','ekflx','ekflxin','ekflxout']):
                    self.rad[varname] = dat[0][r_locs]/self.rad['dr']
                else:
                    self.rad[varname] = dat[0][r_locs]/norm
        return

    def get_slice_coord(self,zoom=0,level=0,xyz=[],axis=0):
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**level/self.Nx3,self.x3max/2**level/self.Nx3]
        x,y,z,dx,dy,dz=self.uni_coord(level=level,xyz=xyz)
        return np.average(x,axis=axis),np.average(y,axis=axis),np.average(z,axis=axis),xyz

    def get_slice(self,var='dens',zoom=0,level=0,xyz=[],axis=0):
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**level/self.Nx3,self.x3max/2**level/self.Nx3]
        return np.average(self.uni_data(var,level=level,xyz=xyz),axis=axis),xyz

    def set_slice(self,varl=['dens','temp'],varsuf='',zoom=0,level=0,xyz=[],axis=0,redo=False):
        for var in varl:
            varname = var+varsuf
            if (redo or varname not in self.slice.keys()):
                self.slice[varname] = {}
                data = self.get_slice(var,zoom=zoom,level=level,xyz=xyz,axis=axis)
                self.slice[varname]['dat'] = data[0]
                self.slice[varname]['xyz'] = data[1]

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
                np.nan_to_num(data, copy=False, posinf=0.0, neginf=0.0)
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
                    else:
                        dats[i] = self.data(var)
                    np.nan_to_num(dats[i], copy=False, posinf=0.0, neginf=0.0)
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

    # TODO(@mhguo): we should have the ability to plot any direction at any position
    def plot_slice(self,var='dens',data=None,varname='',zoom=0,level=0,xyz=[],unit=1.0,bins=None,\
                   title='',label='',xlabel='X',ylabel='Y',cmap='viridis',\
                   norm='log',save=False,figdir='../figure/Simu_',figpath=None,\
                   savepath='',savelabel='',figlabel='',dpi=200,vec=None,stream=None,circle=True,\
                   fig=None,ax=None,xyunit=1.0,colorbar=True,returnim=False,stream_color='k',stream_linewidth=None,\
                   stream_arrowsize=None,vecx='velx',vecy='vely',vel_method='ave',axis=0,**kwargs):
        fig=plt.figure(dpi=dpi) if fig is None else fig
        ax = plt.axes() if ax is None else ax
        bins=int(np.min([self.Nx1,self.Nx2,self.Nx3])) if not bins else bins
        varname = var+f'_{zoom}' if not varname else varname
        if (not xyz):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                   self.x2min/2**zoom,self.x2max/2**zoom,
                   self.x3min/2**level/self.Nx3,self.x3max/2**level/self.Nx3]
        if (data is not None):
            slc=data['dat']*unit
            xyz = list(data['xyz'])
        elif varname in self.slice.keys():
            slc = self.slice[varname]['dat']*unit
            xyz = list(self.slice[varname]['xyz'])
        else:
            slc = self.get_slice(var,zoom=zoom,level=level,xyz=xyz,axis=axis)[0]*unit
        x0,x1,y0,y1,z0,z1 = xyz[0],xyz[1],xyz[2],xyz[3],xyz[4],xyz[5]
        if (vec is not None):
            x,y,z = self.get_slice_coord(zoom=zoom,level=vec,xyz=list(xyz),axis=axis)[:3]
            if (f'{vecx}_{zoom}' in self.slice.keys()):
                u = self.slice[f'{vecx}_{zoom}']['dat']
            else:
                u = self.get_slice(vecx,zoom=zoom,level=level,xyz=list(xyz),axis=axis)[0]
            if (f'{vecy}_{zoom}' in self.slice.keys()):
                v = self.slice[f'{vecy}_{zoom}']['dat']
            else:
                v = self.get_slice(vecy,zoom=zoom,level=level,xyz=list(xyz),axis=axis)[0]
            #x = self.get_slice('x',zoom=zoom,level=vel,xyz=xyz)[0]
            #y = self.get_slice('y',zoom=zoom,level=vel,xyz=xyz)[0]
            fac=max(int(2**(level-vec)),1)
            if (vel_method=='ave'):
                n0,n1=int(u.shape[0]/fac),int(u.shape[1]/fac)
                u=np.average(u.reshape(n0,fac,n1,fac),axis=(1,3))
                n0,n1=int(v.shape[0]/fac),int(v.shape[1]/fac)
                v=np.average(v.reshape(n0,fac,n1,fac),axis=(1,3))
            else:
                u=u[int(fac/2)::fac,int(fac/2)::fac]
                v=v[int(fac/2)::fac,int(fac/2)::fac]
            ax.quiver(x*xyunit, y*xyunit, u, v)
        if (stream is not None):
            x,y,z = self.get_slice_coord(zoom=zoom,level=stream,xyz=xyz,axis=axis)[:3]
            #x = self.get_slice('x',zoom=zoom,level=stream,xyz=xyz)[0]
            #y = self.get_slice('y',zoom=zoom,level=stream,xyz=xyz)[0]
            if (f'{vecx}_{zoom}' in self.slice.keys()):
                u = self.slice[f'{vecx}_{zoom}']['dat']
            else:
                u = self.get_slice(vecx,zoom=zoom,level=level,xyz=list(xyz),axis=axis)[0]
            if (f'{vecy}_{zoom}' in self.slice.keys()):
                v = self.slice[f'{vecy}_{zoom}']['dat']
            else:
                v = self.get_slice(vecy,zoom=zoom,level=level,xyz=list(xyz),axis=axis)[0]
            #x,y=np.meshgrid(x,y)
            #z,x=np.meshgrid(z,x)
            #np.mgrid[-w:w:100j, -w:w:100j]
            #beg=8
            #step=16
            #ax.streamplot(x[beg::step,beg::step], z[beg::step,beg::step], (u[0]/norm[0])[beg::step,beg::step], (v[0]/norm[0])[beg::step,beg::step])
            #print(x.shape,y.shape,u.shape,v.shape)
            # TODO(@mhguo): fix axis=1 case!
            #if(axis==1): x,y = z.swapaxes(0,1), x.swapaxes(0,1)
            if(axis==2): x,y = y,z
            ax.streamplot(x*xyunit, y*xyunit, u, v,color=stream_color,linewidth=stream_linewidth,arrowsize=stream_arrowsize)
        # TODO(@mhguo): fix axis=1 case!
        #if(axis==1): x0,x1,y0,y1 = z0,z1,x0,x1
        if(axis==2): x0,x1,y0,y1 = y0,y1,z0,z1
        im=ax.imshow(slc[::-1,:],extent=(x0*xyunit,x1*xyunit,y0*xyunit,y1*xyunit),\
            norm=norm,cmap=cmap,**kwargs)
        #im=ax.imshow(slc.swapaxes(0,1)[::-1,:],extent=(x0,x1,y0,y1),norm=norm,cmap=cmap,**kwargs)
        #im=ax.imshow(np.rot90(data),cmap='plasma',norm=LogNorm(0.9e-1,1.1e1),extent=extent)
        if(circle and self.header('problem','r_in')):
            ax.add_patch(plt.Circle((0,0),float(self.header('problem','r_in')),ec='k',fc='#00000000'))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if (title != None): ax.set_title(f"Time = {self.time}" if not title else title)
        if (colorbar):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        if (save):
            figpath=figdir+Path(self.path).parts[-1]+'/'+self.label+"/" if not figpath else figpath
            if (not os.path.isdir(figpath)):
                os.mkdir(figpath)
            fig.savefig(f"{figpath}fig_{varname+figlabel if not savelabel else savelabel}_{self.num:04d}.png"\
                        if not savepath else savepath, bbox_inches='tight')
        if (returnim):
            return fig,im
        return fig

    def plot_phase(self,varname='dens_temp',title='',label='',xlabel='X',ylabel='Y',unit=1.0,cmap='viridis',\
                   norm='log',extent=None,density=False,save=False,colorbar=True,savepath='',figdir='../figure/Simu_',\
                   figpath='',x=None,y=None,xshift=0.0,xunit=1.0,yshift=0.0,yunit=1.0,fig=None,ax=None,dpi=128,aspect='auto',**kwargs):
        fig=plt.figure(dpi=dpi) if fig is None else fig
        ax = plt.axes() if ax is None else ax
        dat = self.dist2d[varname]
        extent = [dat['loc1'].min(),dat['loc1'].max(),dat['loc2'].min(),dat['loc2'].max()] if extent is None else extent
        if (density):
            unit /= (extent[1]-extent[0])*(extent[3]-extent[2])/((dat['loc1'].shape[0]-1)*(dat['loc2'].shape[0]-1))
        #im = ax.imshow(dat['dat'].swapaxes(0,1)[::-1,:]*unit,extent=extent,norm=norm,cmap=cmap,aspect=aspect,**kwargs)
        x =  dat['loc1']*xunit+xshift if x is None else x
        y =  dat['loc2']*yunit+yshift if y is None else y
        im = ax.pcolormesh(x,y,dat['dat'].swapaxes(0,1)[:,:]*unit,norm=norm,cmap=cmap,**kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if (title != None): ax.set_title(f"Time = {self.time}" if not title else title)
        if (colorbar):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        if (save):
            figpath=figdir+Path(self.path).parts[-1]+'/'+self.label+"/" if not figpath else figpath
            if not os.path.isdir(figpath):
                os.mkdir(figpath)
            fig.savefig(f"{figpath}fig_{varname}_{self.num:04d}.png"\
                        if not savepath else savepath, bbox_inches='tight')
        return fig


class AthenaDataSet:
    def __init__(self,nlim=10001):
        self.nlim=nlim
        self.ads=[None]*self.nlim
        self._config_func()
        return

    def _config_func(self):
        ad_methods = [method_name for method_name in dir(AthenaData) if callable(getattr(AthenaData, method_name))]
        for method_name in ad_methods:
            self.__dict__[method_name] = lambda *args, **kwargs: [getattr(ad, method_name)(*args, **kwargs) for ad in self.ads]
        return

    def load(self,ilist,info=False,redo=False,**kwargs):
        for i in ilist:
            if(redo or i not in self.alist):
                if(info): print("load:",i)
                if self.ads[i] is None:
                    self.ads[i]=AthenaData(path=self.path,label=self.label,num=i,version=self.version)
                self.ads[i].load(self.binarypath+f"{i:05d}.bin",**kwargs)
            self.alist=sorted(list(set(self.alist + list([i]))))
        return
