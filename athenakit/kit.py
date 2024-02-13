import os
from time import sleep
import numpy as np
from . import macros
if (macros.cupy_enabled):
    import cupy as xp
else:
    xp = np
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d as sp_interp1d

from . import io
from .units import *

mu=0.618
unit=Units(lunit=kpc_cgs,munit=mu*atomic_mass_unit_cgs*kpc_cgs**3,mu=mu)

def zeros_like(obj):
    if (type(obj) is dict):
        return {k:zeros_like(v) for k,v in obj.items()}
    if (type(obj) is list):
        return [zeros_like(a) for a in obj]
    return np.zeros_like(obj)

def plus(a,b):
    if (type(a) is dict):
        return {k:plus(a[k],b[k]) for k in a.keys()}
    if (type(a) is list):
        return [plus(a[i],b[i]) for i in range(len(a))]
    return a+b

def times(a,b):
    if (type(a) is dict):
        return {k:times(a[k],b) for k in a.keys()}
    if (type(a) is list):
        return [times(a[i],b) for i in range(len(a))]
    return a*b

# Convert all binary files in binary path to athdf files in athdf path
def bin_to_athdf(binary_fname,athdf_fname):
    xdmf_fname = athdf_fname + ".xdmf"
    filedata = io.read_binary(binary_fname)
    io.write_athdf(athdf_fname, filedata)
    io.write_xdmf_for(xdmf_fname, os.path.basename(athdf_fname), filedata)
    return

def bins_to_athdfs(binpath,athdfpath,overwrite=False,info=True):
    if not os.path.isdir(athdfpath):
        os.mkdir(athdfpath)
    for file in sorted(os.listdir(binpath)):
        if file.endswith(".bin"):
            binary_fname = os.path.join(binpath, file)
            athdf_fname = os.path.join(athdfpath, file.replace(".bin", ".athdf"))
            xdmf_fname = athdf_fname + ".xdmf"
            if (overwrite or not os.path.exists(athdf_fname) or not os.path.exists(xdmf_fname)):
                if (info): print(f"Converting {file}")
                filedata = io.read_binary(binary_fname)
                io.write_athdf(athdf_fname, filedata)
                io.write_xdmf_for(xdmf_fname, os.path.basename(athdf_fname), filedata)
            else:
                if (info): print(f"Skipping {file}")
    return

@np.vectorize
def CoolFnShure_vec(T):
    if(np.isnan(T)):
        return 0.0
    # original data from Shure et al. paper, covers 4.12 < logt < 8.16
    lhd = [
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928]
    #  for temperatures less than 10^4 K, use Koyama & Inutsuka
    logt=np.log10(T)
    if (logt <= 4.2):
        temp = pow(10.0, logt)
        return (2.0e-19*np.exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*np.sqrt(temp)*np.exp(-92.0/temp))
    
    #for temperatures above 10^8.15 use CGOLS fit
    if (logt > 8.15): return pow(10.0, (0.45*logt - 26.065))
    
    # in between values of 4.2 < log(T) < 8.15
    # linear interpolation of tabulated SPEX cooling rate
    
    ipps = int(25.0*logt) - 103
    ipps = ipps if (ipps < 100) else 100
    ipps = ipps if (ipps > 0  ) else 0
    
    x0    = 4.12 + 0.04*ipps
    dx    = logt - x0
    tcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04))*25.0
    return pow(10.0,tcool)

def CoolFnShure_numpy(T):
    T=np.asarray(T)
    # original data from Shure et al. paper, covers 4.12 < logt < 8.16
    log_t_tab = np.linspace(4.12,8.16,102,endpoint=True)
    log_c_tab = [
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928]
    logt2logc = sp_interp1d(log_t_tab,log_c_tab)
    logt = np.log10(T)
    cool_rate = np.zeros(T.shape)
    loc_c = logt <= 4.2
    loc_h = logt > 8.15
    loc_w = np.logical_not(np.logical_or(loc_c,loc_h))
    #  for temperatures less than 10^4 K, use Koyama & Inutsuka
    temp = T[loc_c]
    cool_rate[loc_c] = 2.0e-19*np.exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*np.sqrt(temp)*np.exp(-92.0/temp)
    #for temperatures above 10^8.15 use CGOLS fit
    cool_rate[loc_h] = 10.0**(0.45*logt[loc_h] - 26.065)
    # in between values of 4.2 < log(T) < 8.15
    # linear interpolation of tabulated SPEX cooling rate
    cool_rate[loc_w] = 10.0**logt2logc(logt[loc_w]) 
    return cool_rate

def CoolFnShure(T):
    if (type(T) is not xp.ndarray):
        return CoolFnShure_numpy(T)
    T=xp.asarray(T)
    # original data from Shure et al. paper, covers 4.12 < logt < 8.16
    log_t_tab = xp.linspace(4.12,8.16,102,endpoint=True)
    log_c_tab = xp.asarray([
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928])
    logt = xp.log10(T)
    cool_rate = xp.zeros(T.shape)
    loc_c = logt <= 4.2
    loc_h = logt > 8.15
    loc_w = xp.logical_not(xp.logical_or(loc_c,loc_h))
    #  for temperatures less than 10^4 K, use Koyama & Inutsuka
    temp = T[loc_c]
    cool_rate[loc_c] = 2.0e-19*xp.exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*xp.sqrt(temp)*xp.exp(-92.0/temp)
    #for temperatures above 10^8.15 use CGOLS fit
    cool_rate[loc_h] = 10.0**(0.45*logt[loc_h] - 26.065)
    # in between values of 4.2 < log(T) < 8.15
    # linear interpolation of tabulated SPEX cooling rate
    cool_rate[loc_w] = 10.0**xp.interp(logt[loc_w],log_t_tab,log_c_tab)
    return cool_rate

# profile solver
def RK4(func,x,y,h):
    k1=func(x,y)
    x+=0.5*h
    k2=func(x,y+0.5*k1*h)
    k3=func(x,y+0.5*k2*h)
    x+=0.5*h
    k4=func(x,y+k3*h)
    y+=1/6*(k1+2*k2+2*k3+k4)*h
    return y
def NFWMass(r,ms,rs):
    return ms*(np.log(1+r/rs)-r/(rs+r))
def NFWDens(r,ms,rs):
    return ms/(4*np.pi*rs**3)/(r/rs*(1+r/rs)**2)

##########################################################################################
## cooling time as a function of temperature and density
##########################################################################################
def rho_T_t_cool(cooling_rho=np.logspace(-4,4,400),cooling_temp=np.logspace(0,8,400)):
    cooling_rho,cooling_temp=np.meshgrid(cooling_rho,cooling_temp)
    cooling_tcool=k_boltzmann_cgs*cooling_temp/cooling_rho/CoolFnShure(cooling_temp)/(gamma-1)/myr_cgs
    return cooling_rho,cooling_temp,cooling_tcool

##########################################################################################
## Some useful functions
##########################################################################################
def smooth(x,s=3,mode='nearest',**kwargs):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x,s,mode=mode,**kwargs)

##########################################################################################
## Scipy Measurements Label with boundary correction
##########################################################################################
import scipy.ndimage as sn

default_struct = sn.generate_binary_structure(3,3)

def clean_tuples(tuples):
    return sorted(set([(min(pair),max(pair)) for pair in tuples]))

def merge_tuples_unionfind(tuples):
    # use classic algorithms union find with path compression
    # https://enp.wikipedia.org/wiki/Disjoint-set_data_structure
    parent_dict = {}

    def subfind(x):
        # update roots while visiting parents 
        if parent_dict[x] != x:
            parent_dict[x] = subfind(parent_dict[x])
        return parent_dict[x]

    def find(x):
        if x not in parent_dict:
            # x forms new set and becomes a root
            parent_dict[x] = x
            return x
        if parent_dict[x] != x:
            # follow chain of parents of parents to find root 
            parent_dict[x] = subfind(parent_dict[x])
        return parent_dict[x]

    # each tuple represents a connection between two items 
    # so merge them by setting root to be the lower root. 
    for p0,p1 in list(tuples):
        r0 = find(p0)
        r1 = find(p1)
        if r0 < r1:
            parent_dict[r1] = r0
        elif r1 < r0:
            parent_dict[r0] = r1

    # for unique parents, subfind the root, replace occurrences with root
    vs = set(parent_dict.values())
    for parent in vs:
        sp = subfind(parent)
        if sp != parent:
            for key in parent_dict:
                if parent_dict[key] == parent:
                    parent_dict[key] = sp

    return parent_dict

def make_dict(mask,struct,boundary,bargs):
    label,things = sn.label(mask,structure=struct)
    cs = clean_tuples(boundary(label,bargs))
    slc = sn.labeled_comprehension(mask,label,range(1,things+1),
                                   lambda a,b: b,
                                   list,
                                   None,
                                   pass_positions=True)
    outdict = dict(zip(range(1,things+1),slc))
    ownerof = merge_tuples_unionfind(cs)
    for key in ownerof:
        if key != ownerof[key]:
            # add key to its owner and remove key
            outdict[ownerof[key]] = np.append(outdict[ownerof[key]],outdict[key])
            outdict.pop(key)
    return outdict,ownerof

def shear_periodic(label,axis,cell_shear,shear_axis):
    # just return the tuple of the one axis
    # 1. get faces
    dim = label.ndim
    size = label.shape[axis]
    select1 = [slice(None)]*dim
    select2 = [slice(None)]*dim
    select1[axis] = 0
    select2[axis] = size-1
    lf1 = label[tuple(select1)]
    lf2 = label[tuple(select2)]
    # 2. now cell shear
    axes = list(range(dim))
    axes.remove(axis)
    aisa = axes.index(shear_axis)
    lf2 = np.roll(lf2,cell_shear,axis=aisa)
    return connect_faces(lf1,lf2)

def periodic(label,axis):
    dim = label.ndim
    size = label.shape[axis]
    select1 = [slice(None)]*dim
    select2 = [slice(None)]*dim
    select1[axis] = 0
    select2[axis] = size-1
    lf1 = label[tuple(select1)]
    lf2 = label[tuple(select2)]
    return connect_faces(lf1,lf2)

def tigress_shear(label,cell_shear):
    # open in Z
    # periodic in Y, so axis = 1
    connectset = set()
    connectset = connectset.union(periodic(label,1))
    # shear periodic in X, so axis = 0, shear_axis = 1
    connectset = connectset.union(shear_periodic(label,0,cell_shear,1))
    return connectset

def tigress(label,cell_shear):
    # open in Z
    # periodic in Y, so axis = 1
    connectset = set()
    connectset = connectset.union(periodic(label,0))
    connectset = connectset.union(periodic(label,1))
    connectset = connectset.union(periodic(label,2))
    return connectset

def tigress_nob(label,cell_shear):
    # open in Z
    # periodic in Y, so axis = 1
    connectset = set()
    return connectset

def connect_faces_simple(lf1,lf2):
    # lf1 and lf2 are label faces
    select = lf1*lf2 > 0
    stack = np.zeros(list(lf1.shape)+[2])
    stack[:,:,0] = lf1
    stack[:,:,1] = lf2
    pairs = stack[select]
    return set([tuple(pair) for pair in pairs])

def connect_faces_rank(lf1,lf2):

    stack = np.zeros([2]+list(lf1.shape))
    stack[0] = lf1
    stack[1] = lf2
    label,things = sn.label(stack > 0,structure=default_struct)
    if things == 0:
        return set()
    slc = sn.labeled_comprehension(stack,label,range(1,things+1),
                                   lambda a: list(set(a)),
                                   list,
                                   None,
                                   pass_positions=False)
    tuples = []
    for region in slc:
        if len(region) == 0:
            continue
        owner = np.min(region)
        for cell in region:
            if cell != owner:
                tuples += [(owner,cell)]
    return set(tuples)

connect_faces = connect_faces_rank
##########################################################################################
## plots
##########################################################################################
def ave(a,n):
    end = -(a.size%n) if(-(a.size%n)) else None
    return np.average((a.ravel()[:end]).reshape(-1,n),axis=1)
##########################################################################################
## plots
##########################################################################################
def mgcolors(name='default'):
    mgcolors=['#3369E8','#009925','#FBBC05','#EA4335',]
    if (name == 'default'):
        return mgcolors
    else:
        return mgcolors

def colors(n,cmap='nipy_spectral',x1=0.0,x2=0.88,beta=0.99):
    colors = ['k','darkviolet',  'b', 'royalblue', 'c', 'g', 'springgreen', 'gold', 'y',  
          'salmon','pink', 'r','darkred', 'm', 'violet',]
    clmap = plt.cm.nipy_spectral  # define the colormap
    clmap = plt.get_cmap(cmap)
    #cmap = plt.cm.terrain  # define the colormap
    #cmap = plt.cm.viridis  # define the colormap
    colors=[tuple(np.array(clmap(x1+i/max(1,n-1)*(x2-x1)))*beta) for i in range(n)]
    return colors
def figure(nrows=1,ncols=1,figsize=(6.4,4.8),dpi=120,sharex=True,squeeze=False,\
    constrained_layout=False,top=0.94, bottom=0.1,left=0.125, right=0.9, wspace=0.02, hspace=0.0):
    fig, axes = plt.subplots(nrows,ncols,figsize=figsize,dpi=dpi,sharex=sharex,\
                         constrained_layout=constrained_layout,squeeze=squeeze)
    fig.subplots_adjust(top=top,bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
    #fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    for ax in axes.flat:
        ax.grid(linestyle='--')
        ax.tick_params(top=True,right=True,which='both',direction="in")
    if(ncols>1):
        for ax in axes[:,-1]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
    return fig,ax
def subplots(nrows=2,ncols=2,figsize=(7.2,5.0),dpi=120,sharex=False,squeeze=False,\
    constrained_layout=False,top=0.94, bottom=0.1,left=0.125, right=0.9, 
    wspace=0.02, hspace=0.0,raw=False,**kwargs):
    fig, axes = plt.subplots(nrows,ncols,figsize=figsize,dpi=dpi,sharex=sharex,\
                         constrained_layout=constrained_layout,squeeze=squeeze,**kwargs)
    fig.subplots_adjust(top=top,bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
    if (raw): return fig,axes
    #fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    if(ncols>1):
        for ax in axes[:,-1]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
    for ax in axes.flat:
        ax.grid(linestyle='--')
        ax.tick_params(bottom=True,top=True,left=True,right=True,which='both',direction="in")
    return fig,axes
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = clr.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# generate a rgb image showing different species
def get_rgb(x,c=[[1,0,0],[0.98,0.6,0.02],[0.0,0.6,0.1],[0.05,0.1,1.0]]):
    u,v,w=0.,0.,0.
    for i in range(len(x)):
        u=u+c[i][0]*x[i]
        v=v+c[i][1]*x[i]
        w=w+c[i][2]*x[i]
    return np.array([u,v,w]).transpose([1,2,0])
