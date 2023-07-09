import os
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d as sp_interp1d

from . import io

# units
cm_cgs = 1.0;                           # cm
pc_cgs = 3.0856775809623245e+18;        # cm
kpc_cgs = 3.0856775809623245e+21;       # cm
g_cgs = 1.0;                            # g
msun_cgs = 1.98841586e+33;              # g
atomic_mass_unit_cgs = 1.660538921e-24; # g
s_cgs = 1.0;                            # s
yr_cgs = 3.15576e+7;                    # s
myr_cgs = 3.15576e+13;                  # s
cm_s_cgs = 1.0;                         # cm/s
km_s_cgs = 1.0e5;                       # cm/s
g_cm3_cgs = 1.0;                        # g/cm^3
erg_cgs = 1.0;                          # erg
dyne_cm2_cgs = 1.0;                     # dyne/cm^2  
kelvin_cgs = 1.0;                       # k
k_boltzmann_cgs = 1.3806488e-16;        # erg/k
grav_constant_cgs = 6.67408e-8;         # cm^3/(g*s^2)
speed_of_light_cgs = 2.99792458e10      # cm/s

class Units:
    def __init__(self,lunit=pc_cgs,munit=atomic_mass_unit_cgs*pc_cgs**3,tunit=myr_cgs,mu=1.0):
        self.length_cgs=lunit
        self.mass_cgs=munit
        self.time_cgs=tunit
        self.mu=mu
    @property
    def velocity_cgs(self):
        return self.length_cgs/self.time_cgs
    @property
    def density_cgs(self):
        return self.mass_cgs/self.length_cgs**3
    @property
    def energy_cgs(self):
        return self.mass_cgs*self.velocity_cgs**2
    @property
    def pressure_cgs(self):
        return self.energy_cgs/self.length_cgs**3
    @property
    def temperature_cgs(self):
        return self.velocity_cgs**2*self.mu*atomic_mass_unit_cgs/k_boltzmann_cgs
    @property
    def grav_constant(self):
        return grav_constant_cgs*self.density_cgs*self.time_cgs**2
    @property
    def speed_of_light(self):
        return speed_of_light_cgs/self.velocity_cgs
    @property
    def number_density_cgs(self):
        return self.density_cgs/self.mu/atomic_mass_unit_cgs
    @property
    def cooling_cgs(self):
        return self.pressure_cgs/self.time_cgs/self.number_density_cgs**2
    @property
    def heating_cgs(self):
        return self.pressure_cgs/self.time_cgs/self.number_density_cgs
    @property
    def conductivity_cgs(self):
        return self.pressure_cgs*self.velocity_cgs*self.length_cgs/self.temperature_cgs
    @property
    def entropy_kevcm2(self):
        kev_erg=1.60218e-09
        gamma=5./3.
        return self.pressure_cgs/kev_erg/self.number_density_cgs**gamma
    @property
    def magnetic_field_cgs(self):
        return np.sqrt(4.0*np.pi*self.density_cgs)*self.velocity_cgs

mu=0.618
unit=Units(lunit=kpc_cgs,munit=mu*atomic_mass_unit_cgs*kpc_cgs**3,mu=mu)

# Convert all binary files in binary path to athdf files in athdf path
def bin_to_athdf(binpath,athdfpath,overwrite=False):
    if not os.path.isdir(athdfpath):
        os.mkdir(athdfpath)
    for file in sorted(os.listdir(binpath)):
        if file.endswith(".bin"):
            binary_fname = os.path.join(binpath, file)
            athdf_fname = os.path.join(athdfpath, file.replace(".bin", ".athdf"))
            xdmf_fname = athdf_fname + ".xdmf"
            if (overwrite or not os.path.exists(athdf_fname) or not os.path.exists(xdmf_fname)):
                print(f"Converting {file}")
                filedata = io.read_binary(binary_fname)
                io.write_athdf(athdf_fname, filedata)
                io.write_xdmf_for(xdmf_fname, os.path.basename(athdf_fname), filedata)
            else:
                print(f"Skipping {file}")
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

def CoolFnShure(T):
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

# parameters
gamma       = 5/3     # gamma
potential   = 1       # turn on potential
m_bh        = 4.3e2   # BH mass: 6.5e9 Msun # Mass unit: 1.516e7 Msun
r_in        = 0.0     # inner radius: 100 pc
m_star      = 2e4     # 3e11 Msun # Mass unit: 1.516e7 Msun
r_star      = 2.0     # 2kpc
m_dm        = 2.0e6   # DM Mass: 3.0e13 Msun # Mass unit: 1.516e7 Msun
r_dm        = 60.0    # 60kpc
sink_d      = 1e-2    # density of sink cells
sink_t      = 1e-1    # eint/dens=1/gm1*T: temperature of sink cells
rad_entry   = 2.0
dens_entry  = 1e-1
#k_0         = 1.0
#xi          = 1.75
k_0         = 1.1
xi          = 1.1

# profile solver
def NFWMass(r,ms,rs):
    return ms*(np.log(1+r/rs)-r/(rs+r))
def TotMass(r,m=m_bh,mc=m_star,rc=r_star,ms=m_dm,rs=r_dm):
    return m+NFWMass(r,mc,rc)+NFWMass(r,ms,rs)
def Acceleration(r,m,mc,rc,ms,rs,g):
    return -g*(TotMass(r,m,mc,rc,ms,rs))/r**2
def DrhoDr(x,rho):
    r = x*rad_entry
    accel = rad_entry*Acceleration(r,m_bh,m_star,r_star,m_dm,r_dm,unit.grav_constant)
    #print(accel)
    #print(rho,gamma,xi)
    grad = (2*rho**(2-gamma)*accel/k_0-rho*xi*x**(xi-1))/((1+x**xi)*gamma)
    return grad
def RK4(func,x,y,h):
    k1=func(x,y)
    x+=0.5*h
    k2=func(x,y+0.5*k1*h)
    k3=func(x,y+0.5*k2*h)
    x+=0.5*h
    k4=func(x,y+k3*h)
    y+=1/6*(k1+2*k2+2*k3+k4)*h
    return y
def SolveDens(N=2048,logh=0.002):
    N2=int(N/2)
    dens_arr = np.zeros(N)
    dens = dens_entry
    dens_arr[N2]=dens
    for i in range(N2):
        x = 10**(-i*logh)
        h = 10**(-(i+1)*logh)-x
        dens = RK4(DrhoDr,x,dens,h)
        dens_arr[N2-i-1]=dens
    dens = dens_entry
    for i in range(N2-1):
        x = 10**(i*logh)
        h = 10**((i+1)*logh)-x
        dens = RK4(DrhoDr,x,dens,h)
        dens_arr[N2+i+1]=dens
    xs=np.logspace(-logh*N2,logh*(N2-1),N,endpoint=True)
    pres_arr = 0.5*k_0*(1.0+pow(xs,xi))*pow(dens_arr,gamma)
    rss=dict()
    rss['r']=xs*rad_entry
    rss['dens']=dens_arr
    rss['pres']=pres_arr
    rss['temp']=rss['pres']/rss['dens']
    rss['entropy']=rss['pres']/rss['dens']**(5/3)
    return rss

# solve
ran=SolveDens(N=12000)
ran['mass']=TotMass(ran['r'],m_bh,m_star,r_star,m_dm,r_dm)
ran['g']=Acceleration(ran['r'],m_bh,m_star,r_star,m_dm,r_dm,unit.grav_constant)
ran['t_ff']=np.pi/4.0*np.sqrt(2.*ran['r']/-ran['g'])
ran['v_ff']=np.sqrt(2.*ran['r']*-ran['g'])
ran['v_kep']=np.sqrt(ran['r']*-ran['g'])
ran['Omega']=ran['v_kep']/ran['r']
ran['am_kep']=np.sqrt(ran['r']**3*-ran['g'])
ran['potential']=ran['r']*-ran['g']
ran['r_B']=unit.grav_constant*m_bh/(gamma*ran['temp'])
ran['Mdot_B']=np.pi*(unit.grav_constant*m_bh)**2*ran['dens']/(gamma*ran['temp'])**1.5

##########################################################################################
## cooling time as a function of temperature and density
##########################################################################################
def rho_T_t_cool(cooling_rho=np.logspace(-4,4,400),cooling_temp=np.logspace(0,8,400)):
    cooling_rho,cooling_temp=np.meshgrid(cooling_rho,cooling_temp)
    cooling_tcool=k_boltzmann_cgs*cooling_temp/cooling_rho/CoolFnShure(cooling_temp)/(gamma-1)/myr_cgs
    return cooling_rho,cooling_temp,cooling_tcool

##########################################################################################
## SNR
##########################################################################################

# analytical function
class SNR:
    # TODO(@mhguo): add units!!!
    def __init__(self,n=1,M=3,E=1,mu=0.618,config=True):
        #E [erg]
        #n cm^-3
        #M Msun
        self.n0=1
        self.M0=msun_cgs
        self.E0=1e51
        self.mu0=1.4
        self.mu=mu
        self.unit=Units(lunit=pc_cgs,munit=mu*atomic_mass_unit_cgs*pc_cgs**3,tunit=myr_cgs,mu=mu)
        self.n=n*self.n0/self.unit.number_density_cgs
        self.M=M*self.M0/self.unit.mass_cgs
        self.E=E*self.E0/self.unit.energy_cgs
        self.Ei=0.72*self.E
        self.Ek=0.28*self.E
        self.v_free=np.sqrt(2*self.E/self.M)
        self.r_free=(self.M/(4/3*np.pi*self.n))**(1/3)
        self.t_free=self.r_free/self.v_free

        #self.t_free=4.64e-4*E**(-1/2)*n**(-1/3)
        #self.r_free=2.75*n**(-1/3)
        
        self.epsilon=1.15167
        self.t_sf=0.044*E**0.22*n**-0.55/(self.mu/self.mu0)**-0.55
        self.r_sf=self._r_st(self.t_sf)
        self.v_sf=self._v_st(self.t_sf)
        self.mom_sf=2.69*self.n*self.v_sf*self.r_sf**3
        #self.r_sf=22.6*E**0.29*n**-0.42/(self.mu/self.mu0)**-0.42
        #self.mom_sf=2.17e5*E**0.93*n**-0.13/(self.mu/self.mu0)**-0.13
        self.evo={}
        if(config): self.config()
        return
    
    def _r_st(self,t):
        return self.epsilon*self.E**(1/5)*self.n**(-1/5)*t**(2/5)
    def _v_st(self,t):
        return 2/5*self.epsilon*self.E**(1/5)*self.n**(-1/5)*t**(-3/5)
    @np.vectorize
    def _r(self,t):
        if(t<=self.t_free):
            return np.sqrt(2*self.E/self.M)*t
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return self._r_st(t)
        else:
            return self.r_sf*(t/self.t_sf)**(2/7)
        #else:
        #    return 30*(t/0.1)**(2/7)
    @np.vectorize
    def _v(self,t):
        if(t<=self.t_free):
            return np.sqrt(2*self.E/self.M)
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return self._v_st(t)
        else:
            return 2/7*self.r_sf/self.t_sf*(t/self.t_sf)**(-5/7)
    @np.vectorize
    def _momr(self,t):
        if(t<=self.t_free):
            return self.M*np.sqrt(2*self.E/self.M)
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return 2.69*self.n*self._v_st(t)*self._r_st(t)**3
        else:
            return self.mom_sf*(1+4.6*((t/self.t_sf)**(1/7)-1.0))
    @np.vectorize
    def _pres(self,t):
        if(t<=self.t_free):
            #return self.M*np.sqrt(2*self.E/self.M)
            return self.Ei/2.0/np.pi/self._r_st(t)**3
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return self.Ei/2.0/np.pi/self._r_st(t)**3
        else:
            return self.Ei/2.0/np.pi/self._r_st(self.t_sf)**3*(t/self.t_sf)**(-10/7)
    def r(self,t):
        return self._r(self,t)
    def v(self,t):
        return self._v(self,t)
    def momr(self,t):
        return self._momr(self,t)
    def pres(self,t):
        return self._pres(self,t)

    def config(self,t=np.logspace(-4,0,300)):
        self.evo['t']=t
        self.evo['r']=self.r(t)
        self.evo['v']=self.v(t)
        self.evo['momr']=self.momr(t)
        self.evo['pres']=self.pres(t)
        self.evo['m']=4/3*np.pi*self.n*self.evo['r']**3+self.M
        self.evo['eta']=self.evo['v']/(self.evo['r']/self.evo['t'])
        self.evo['etot']=np.full(self.evo['t'].shape,self.E)
        self.evo['ei']=0.72*self.evo['etot']
        self.evo['ek']=0.28*self.evo['etot']
        #self.evo['momr']=2.69/(4/3*np.pi)*self.evo['m']*self.evo['v']

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
def subplots(nrows=2,ncols=2,figsize=(7.2,5.0),dpi=120,sharex=True,squeeze=False,\
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
