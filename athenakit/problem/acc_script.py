'''
Script for running analysis of athena++ simulation

Example:
    python -u script.py -p ../simulation -t wpv
'''

import os
import sys
import argparse
import time
import numpy as np
try:
    import cupy as xp
except ImportError:
    import numpy as xp
from matplotlib import pyplot as plt
import multiprocessing as mp
from pathlib import Path

sys.path.append('/u/mguo/Git')
import athenaresearch.athenakit as ak
from athenaresearch.athenakit.problem import acc
plt.style.use('~/Git/pykit/pykit/mplstyle/mg')

def adwork(ad,zlist=None,bins=256):
    acc.add_tools(ad)
    acc.add_tran(ad)
    acc.add_data(ad)
    t_cold=ad.header('problem','t_cold',float)
    # TODO: use a correct ran!
    t_initial=xp.interp(ad.data('r'),xp.asarray(ad.rad_initial['r']),xp.asarray(ad.rad_initial['temp']))
    t_hot=ad.header('problem','tf_hot',float)*t_initial

    varl=['vol','dens','temp','velx','vely','velz','etot','ekin','eint','velr','pres','entropy',\
          'mdot','mdotin','mdotout','velin','velout','vtot','vrot','amx','amy','amz','amtot',]
    ad.set_sum(varl)

    jlist=[0,3,5,7,9]
    #jlist=[0,3,5,7,]
    max_level=int(ad.mb_logical[:,-1].max())
    #if(max_level>11): jlist=list(range(max_level-9,max_level,2))
    if(max_level>8): jlist=list(range(max_level-8,max_level+1,2))
    jlist=[0]+list(range(max_level-8,max_level+1,2))
    if (zlist is not None): jlist=[int(z) for z in zlist]
    print('slice '+str(ad.num))
    for j in jlist:
        #ad.set_slice(varl=varl,zoom=j,level=j+1)
        zoom=j
        level=j+1
        varl=['dens','temp','velx','vely','bccx','bccy','pres','vtot',]
        ad.set_slice(varl=varl,key=f'z_{j}',level=level,axis='z',
                     xyz=[ad.x1min/2**zoom,ad.x1max/2**zoom,
                          ad.x2min/2**zoom,ad.x2max/2**zoom,
                          ad.x3min/2**level/ad.Nx3,ad.x3max/2**level/ad.Nx3])
        varl=['dens','temp','vely','velz','bccy','bccz','pres','vtot',]
        ad.set_slice(varl=varl,key=f'x_{j}',level=level,axis='x',
                     xyz=[ad.x1min/2**level/ad.Nx1,ad.x1max/2**level/ad.Nx1,
                          ad.x2min/2**zoom,ad.x2max/2**zoom,
                          ad.x3min/2**zoom,ad.x3max/2**zoom])
        varl=['dens','temp','velx','velz','bccx','bccz','pres','vtot',]
        ad.set_slice(varl=varl,key=f'y_{j}',level=level,axis='y',
                     xyz=[ad.x1min/2**zoom,ad.x1max/2**zoom,
                          ad.x2min/2**zoom/ad.Nx2,ad.x2max/2**zoom/ad.Nx2,
                          ad.x3min/2**zoom,ad.x3max/2**zoom])
        varl=['dens',]
        ad.set_profile2d(['x','y'],varl,key=f'x-y_{j}',weights='vol',bins=128,where=xp.abs(ad.data('z'))<ad.x3max/2**zoom,
                 range=[[ad.x1min/2**zoom,ad.x1max/2**zoom],[ad.x2min/2**zoom,ad.x2max/2**zoom]])
        ad.set_profile2d(['y','z'],varl,key=f'y-z_{j}',weights='vol',bins=128,where=xp.abs(ad.data('x'))<ad.x1max/2**zoom,
                 range=[[ad.x2min/2**zoom,ad.x2max/2**zoom],[ad.x3min/2**zoom,ad.x3max/2**zoom]])
        ad.set_profile2d(['x','z'],varl,key=f'x-z_{j}',weights='vol',bins=128,where=xp.abs(ad.data('y'))<ad.x2max/2**zoom,
                 range=[[ad.x1min/2**zoom,ad.x1max/2**zoom],[ad.x3min/2**zoom,ad.x3max/2**zoom]])
    
    print('phase '+str(ad.num))
    varl=['dens','temp','amx','amy','amz','amtot']
    ad.set_hist(varl,weights='vol')
    ad.set_hist(varl,weights='mass')
    where = ad.data('temp')<t_cold
    if (where.any()):
        ad.set_hist(varl,key='vol_cold',weights='vol',where=where)
        ad.set_hist(varl,key='mass_cold',weights='mass',where=where)
    
    varl=[['r','dens'],['r','temp'],['r','pres']]
    ad.set_hist2d(varl,weights='vol',bins=bins,scales='log')
    varl=[['dens','temp'],['r','dens'],['r','temp'],['r','pres'],['r','c_s'],['r','amtot'],
          ['r','btot'],['r','beta'],['r','vtot'],['r','v_A']]
    ad.set_hist2d(varl,weights='mass',bins=bins,scales='log')

    print('radial '+str(ad.num))
    varl=['mdot','mdotin','mdotout','momdot','momdotin','momdotout','ekdot','ekdotin','ekdotout']
    for suf in ['','_cold','_warm','_hot']:
        ad.set_profile('r',varl=[var+suf for var in varl],key='r_vol',bins=bins,scales='log',weights='vol')

    varl=['dens','temp','velx','vely','velz','etot','ekin','eint','velr','pres','entropy',\
          'velin','velout','vtot','vrot','amx','amy','amz','amtot',\
          'bccx','bccy','bccz','btot^2','beta','v_A']
    ad.set_profile('r',varl=varl,key='r_vol',bins=bins,scales='log',weights='vol',range=[[ad.rmin,ad.rmax]])
    ad.set_profile('r',varl=varl,key='r_mass',bins=bins,scales='log',weights='mass',range=[[ad.rmin,ad.rmax]])
    where = ad.data('temp')<t_cold
    if (where.any()):
        ad.set_profile('r',varl=varl,key='r_vol_cold',bins=bins,scales='log',weights='vol',where=where,range=[[ad.rmin,ad.rmax]])
        ad.set_profile('r',varl=varl,key='r_mass_cold',bins=bins,scales='log',weights='mass',where=where,range=[[ad.rmin,ad.rmax]])
    where = (ad.data('temp')>=t_cold) & (ad.data('temp')<t_hot)
    if (where.any()):
        ad.set_profile('r',varl=varl,key='r_vol_warm',bins=bins,scales='log',weights='vol',where=where,range=[[ad.rmin,ad.rmax]])
        ad.set_profile('r',varl=varl,key='r_mass_warm',bins=bins,scales='log',weights='mass',where=where,range=[[ad.rmin,ad.rmax]])
    where = ad.data('temp')>t_hot
    if (where.any()):
        ad.set_profile('r',varl=varl,key='r_vol_hot',bins=bins,scales='log',weights='vol',where=where,range=[[ad.rmin,ad.rmax]])
        ad.set_profile('r',varl=varl,key='r_mass_hot',bins=bins,scales='log',weights='mass',where=where,range=[[ad.rmin,ad.rmax]])
    
    print('prof2d '+str(ad.num))
    varl=["dens",'temp','pres','btot^2','beta',"tran_velR","tran_velz","tran_velphi","tran_bccR","tran_bccz","tran_bccphi",
          "tran_dens_velR","tran_dens_velz","tran_radial_flow",
          "tran_stress_zphi_hydro","tran_stress_zphi_maxwell","tran_stress_zphi",
          "tran_stress_Rphi_hydro","tran_stress_Rphi_maxwell","tran_stress_Rphi",
          ]
    Rmax=ad.rmax/2**7 # 250 pc
    ad.set_profile2d(['tran_R','tran_z'],varl=varl,key='tRz_vol',bins=128,weights='vol',
                     where=np.logical_and(ad.data('tran_R')<Rmax,np.abs(ad.data('tran_z'))<Rmax))
    ad.set_profile('tran_R',varl=varl,key='tR_vol',bins=256,weights='vol',scales='log',
                   where=(ad.data('tran_R')>ad.rin) & (ad.data('tran_R')<10*Rmax) & (np.abs(ad.data('tran_z/R'))<0.1))
    R_here=5e4 # 16 pc
    ad.set_profile('tran_z',varl=varl,key='tz_vol_R16pc',bins=128,weights='vol',
                   where=(ad.data('tran_R')>0.8*R_here) & (ad.data('tran_R')<1.2*R_here) & (np.abs(ad.data('tran_z'))<2*R_here))
    R_here=5e5 # 160 pc
    ad.set_profile('tran_z',varl=varl,key='tz_vol_R160pc',bins=128,weights='vol',
                   where=(ad.data('tran_R')>0.8*R_here) & (ad.data('tran_R')<1.2*R_here) & (np.abs(ad.data('tran_z'))<2*R_here))

    return ad

def adupdate(ad):
    acc.add_tools(ad)
    acc.add_tran(ad)
    acc.add_data(ad)
    varl=["dens",'temp','pres','btot^2','beta',"tran_velR","tran_velz","tran_velphi","tran_bccR","tran_bccz","tran_bccphi",
          "tran_dens_velR","tran_dens_velz","tran_radial_flow",
          "tran_stress_zphi_hydro","tran_stress_zphi_maxwell","tran_stress_zphi",
          "tran_stress_Rphi_hydro","tran_stress_Rphi_maxwell","tran_stress_Rphi",
          ]
    #Rmax=ad.rmax/2**7 # 250 pc
    #ad.set_profile2d(['tran_R','tran_z/R'],varl=varl,key='tRzR_vol',bins=128,range=[[ad.rin,10*Rmax],[-2.0,2.0]],
    #                 weights='vol',scales=['log','linear'])
    Rmax=ad.rmax/2**10 # 30 pc
    ad.set_profile2d(['tran_R','tran_z'],varl=varl,key='tRz_l10_vol',bins=128,weights='vol',
                     where=np.logical_and(ad.data('tran_R')<Rmax,np.abs(ad.data('tran_z'))<Rmax))
    return ad

def adplot(ad,zlist=None):
    figdir=ad.figpath
    unit=ad.unit
    ran=ad.rad_initial
    lunit,tunit,munit,vunit=unit.length_cgs/ak.pc_cgs,unit.time_cgs/ak.myr_cgs,unit.mass_cgs/ak.msun_cgs,unit.velocity_cgs/ak.km_s_cgs
    lunit=1.0 # for GR
    Tunit,mdot_unit,magnetic_unit=unit.temperature_cgs,unit.mdot_msun_yr,unit.magnetic_field_cgs/1e-6
    #'''
    
    jlist=[0,3,5,7,9]
    #jlist=[0,3,5,7,]
    max_level=int(ad.mb_logical[:,-1].max())
    #if(max_level>11): jlist=list(range(max_level-9,max_level,2))
    if(max_level>8): jlist=list(range(max_level-8,max_level+1,2))
    jlist=[0]+list(range(max_level-8,max_level+1,2))
    if (zlist is not None): jlist=[int(z) for z in zlist]
    fig,axes=ak.subplots(4,6,figsize=(16,10),dpi=128,wspace=0.3,hspace=0.25,bottom=0.08,top=0.92,left=0.05,right=0.95,raw=True)
    fig.suptitle(fr'Time={ad.time*tunit:.2f} Myr (Black lines: $v$, Grey lines: $B$)')
    for i,j in enumerate(jlist):#
        zoom=j
        vec=None
        xyz=None
        stream=True
        level=j+1
        xyunit=lunit
        prof=ad.profs[f'y-z_{j}']
        x,y=prof['edges'].values()
        dxdy=np.matmul(np.diff(x).reshape(-1,1),np.diff(y).reshape(1,-1))
        ad.plot_image(fig=fig,ax=axes[0,i],x=x*xyunit,y=y*xyunit,img=(prof['dens']*prof['norm']/dxdy).T*unit.length_cgs,
            label=r'$N\rm\,[cm^{-2}]$' if (j==jlist[-1]) else None,cmap='magma',xlabel=' ',ylabel='Z [M]' if (i==0) else ' ',
            dpi=128,norm='log',title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[1,i],var='dens',  key=f'x_{j}',axis='x',save=False,label=r'$n\rm\,[cm^{-3}]$' if (j==jlist[-1]) else None,
            cmap='Spectral_r',xlabel=' ',ylabel='Z [M]' if (i==0) else ' ',zoom=zoom,vec=vec,stream=stream,level=level,
            xyz=xyz,vecx='vely',vecy='velz',unit=1,                   xyunit=xyunit,dpi=128,norm='log',vmin=1e-3,vmax=1e5,
            title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[2,i],var='temp',  key=f'x_{j}',axis='x',save=False,label=r'$T\rm\,[K]$' if (j==jlist[-1]) else None,
            cmap='RdYlBu_r',xlabel=' ',ylabel='Z [M]' if (i==0) else ' ',  zoom=zoom,vec=vec,stream=stream,level=level,
            xyz=xyz,vecx='bccy',vecy='bccz',unit=Tunit,xyunit=xyunit,dpi=128,norm='log',vmin=5e4,vmax=5e9,stream_para=dict(color='grey',),
            title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[3,i],var='vtot',key=f'x_{j}',axis='x',save=False,label=r'$|v|\rm\,[km\,s^{-1}]$' if (j==jlist[-1]) else None, 
            cmap='plasma',xlabel='Y [M]',ylabel='Z [M]' if (i==0) else ' ',zoom=zoom,vec=vec,level=level,
            xyz=xyz,unit=unit.velocity_cgs/ak.km_s_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e1,vmax=3e4,
            title=None,aspect='equal',yticklabels=[])
        #ad.plot_slice(fig=fig,ax=axes[1,1],var='pres',  key=f'z_{j}',save=False,label=r'$P\rm\,[k_B\,K\,cm^{-3}]$',
        #    cmap='plasma', xlabel='X [M]',ylabel='',       zoom=zoom,vec=vec,level=level,
        #    xyz=xyz,unit=unit.temperature_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e5,vmax=1e13,
        #    title=f'Time = {ad.time:.2f}',aspect='equal')
        #ad.abins[i].plot_snapshot(N=2**5,varname='dens',save=True)
    savelabel=f'image_x'
    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png")#,bbox_inches='tight'
    plt.close(fig)
    #'''

    #'''
    fig,axes=ak.subplots(4,6,figsize=(16,10),dpi=128,wspace=0.3,hspace=0.25,bottom=0.08,top=0.92,left=0.05,right=0.95,raw=True)
    fig.suptitle(fr'Time={ad.time*tunit:.2f} Myr (Black lines: $v$, Grey lines: $B$)')
    for i,j in enumerate(jlist):#
        zoom=j
        vec=None
        xyz=None
        stream=True
        level=j+1
        xyunit=lunit
        prof=ad.profs[f'x-z_{j}']
        x,y=prof['edges'].values()
        dxdy=np.matmul(np.diff(x).reshape(-1,1),np.diff(y).reshape(1,-1))
        ad.plot_image(fig=fig,ax=axes[0,i],x=x*xyunit,y=y*xyunit,img=(prof['dens']*prof['norm']/dxdy).T*unit.length_cgs,
            label=r'$N\rm\,[cm^{-2}]$' if (j==jlist[-1]) else None,cmap='magma',xlabel=' ',ylabel='Z [M]' if (i==0) else ' ',
            dpi=128,norm='log',title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[1,i],var='dens',  key=f'y_{j}',axis='y',save=False,label=r'$n\rm\,[cm^{-3}]$' if (j==jlist[-1]) else None,
            cmap='Spectral_r',xlabel=' ',ylabel='Z [M]' if (i==0) else ' ',zoom=zoom,vec=vec,stream=stream,level=level,
            xyz=xyz,vecx='velx',vecy='velz',unit=1,                   xyunit=xyunit,dpi=128,norm='log',vmin=1e-3,vmax=1e5,
            title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[2,i],var='temp',  key=f'y_{j}',axis='y',save=False,label=r'$T\rm\,[K]$' if (j==jlist[-1]) else None,
            cmap='RdYlBu_r',xlabel=' ',ylabel='Z [M]' if (i==0) else ' ',  zoom=zoom,vec=vec,stream=stream,level=level,stream_para=dict(color='grey',),
            xyz=xyz,vecx='bccx',vecy='bccz',unit=Tunit,xyunit=xyunit,dpi=128,norm='log',vmin=5e4,vmax=5e9,
            title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[3,i],var='vtot',key=f'y_{j}',axis='y',save=False,label=r'$|v|\rm\,[km\,s^{-1}]$' if (j==jlist[-1]) else None, 
            cmap='plasma',xlabel='X [M]',ylabel='Z [M]' if (i==0) else ' ',zoom=zoom,vec=vec,level=level,
            xyz=xyz,unit=unit.velocity_cgs/ak.km_s_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e1,vmax=3e4,
            title=None,aspect='equal',yticklabels=[])
        #ad.plot_slice(fig=fig,ax=axes[1,1],var='pres',  key=f'z_{j}',save=False,label=r'$P\rm\,[k_B\,K\,cm^{-3}]$',
        #    cmap='plasma', xlabel='X [M]',ylabel='',       zoom=zoom,vec=vec,level=level,
        #    xyz=xyz,unit=unit.temperature_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e5,vmax=1e13,
        #    title=f'Time = {ad.time:.2f}',aspect='equal')
        #ad.abins[i].plot_snapshot(N=2**5,varname='dens',save=True)
    savelabel=f'image_y'
    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png")#,bbox_inches='tight'
    plt.close(fig)
    #'''

    #'''
    fig,axes=ak.subplots(4,6,figsize=(16,10),dpi=128,wspace=0.3,hspace=0.25,bottom=0.08,top=0.92,left=0.05,right=0.95,raw=True)
    fig.suptitle(fr'Time={ad.time*tunit:.2f} Myr (Black lines: $v$, Grey lines: $B$)')
    for i,j in enumerate(jlist):#
        zoom=j
        vec=None
        xyz=None
        stream=True
        level=j+1
        xyunit=lunit
        prof=ad.profs[f'x-y_{j}']
        x,y=prof['edges'].values()
        dxdy=np.matmul(np.diff(x).reshape(-1,1),np.diff(y).reshape(1,-1))
        ad.plot_image(fig=fig,ax=axes[0,i],x=x*xyunit,y=y*xyunit,img=(prof['dens']*prof['norm']/dxdy).T*unit.length_cgs,
            label=r'$N\rm\,[cm^{-2}]$' if (j==jlist[-1]) else None,cmap='magma',xlabel=' ',ylabel='Y [M]' if (i==0) else ' ',
            dpi=128,norm='log',title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[1,i],var='dens',  key=f'z_{j}',axis='z',save=False,label=r'$n\rm\,[cm^{-3}]$' if (j==jlist[-1]) else None,
            cmap='Spectral_r',xlabel=' ',ylabel='Y [M]' if (i==0) else ' ',zoom=zoom,vec=vec,stream=stream,level=level,
            xyz=xyz,vecx='velx',vecy='vely',unit=1,                   xyunit=xyunit,dpi=128,norm='log',vmin=1e-3,vmax=1e5,
            title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[2,i],var='temp',  key=f'z_{j}',axis='z',save=False,label=r'$T\rm\,[K]$' if (j==jlist[-1]) else None,
            cmap='RdYlBu_r',xlabel=' ',ylabel='Y [M]' if (i==0) else ' ',  zoom=zoom,vec=vec,stream=stream,level=level,stream_para=dict(color='grey',),
            xyz=xyz,vecx='bccx',vecy='bccy',unit=Tunit,xyunit=xyunit,dpi=128,norm='log',vmin=5e4,vmax=5e9,
            title=None,aspect='equal',yticklabels=[])
        ad.plot_slice(fig=fig,ax=axes[3,i],var='vtot',key=f'z_{j}',axis='z',save=False,label=r'$|v|\rm\,[km\,s^{-1}]$' if (j==jlist[-1]) else None, 
            cmap='plasma',xlabel='X [M]',ylabel='Y [M]' if (i==0) else ' ',zoom=zoom,vec=vec,level=level,
            xyz=xyz,unit=unit.velocity_cgs/ak.km_s_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e1,vmax=3e4,
            title=None,aspect='equal',yticklabels=[])
        #ad.plot_slice(fig=fig,ax=axes[1,1],var='pres',  key=f'z_{j}',save=False,label=r'$P\rm\,[k_B\,K\,cm^{-3}]$',
        #    cmap='plasma', xlabel='X [M]',ylabel='',       zoom=zoom,vec=vec,level=level,
        #    xyz=xyz,unit=unit.temperature_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e5,vmax=1e13,
        #    title=f'Time = {ad.time:.2f}',aspect='equal')
        #ad.abins[i].plot_snapshot(N=2**5,varname='dens',save=True)
    savelabel=f'image_z'
    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png")#,bbox_inches='tight'
    plt.close(fig)
    #'''

    #'''
    print('phase '+str(ad.num))
    fig,axes=ak.subplots(2,4,figsize=(16,8),dpi=128,wspace=0.5,top=0.88,hspace=0.4,left=0.05,right=0.95,raw=True)
    fig.suptitle(f'Time={ad.time*tunit} Myr')
    savelabel=f'phase_0'
    vmin, vmax = 1e0,1e12
    locs=np.where((ran['r']-ad.rin)*(ran['r']-ad.rmax)<=0)
    key,weights='mass','mass'
    cmap='Spectral_r'

    fig=ad.plot_phase(fig=fig,ax=axes[0,0],key=key,varname='dens-temp',weights=weights,xlabel=r'$n\rm\,[cm^{-3}]$',ylabel=r'$T\,\rm [K]$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,yunit=unit.temperature_cgs,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')

    fig=ad.plot_phase(fig=fig,ax=axes[0,1],key=key,varname='r-dens',weights=weights,xlabel=r'$r\rm\,[M]$',ylabel=r'$n\,\rm [cm^{-3}]$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,xunit=lunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[0,1].plot(ran['r'][locs]*lunit,ran['dens'][locs],':',label=r'$n$',color='k')
    
    fig=ad.plot_phase(fig=fig,ax=axes[1,0],key=key,varname='r-temp',weights=weights,xlabel=r'$r\rm\,[M]$',ylabel=r'$T\,\rm [K]$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,xunit=lunit,yunit=unit.temperature_cgs,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[1,0].plot(ran['r'][locs]*lunit,ran['temp'][locs]*Tunit,':',label=r'$T$',color='k')

    fig=ad.plot_phase(fig=fig,ax=axes[1,1],key=key,varname='r-amtot',weights=weights,xlabel=r'$r\rm\,[M]$',ylabel=r'$L$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,xunit=lunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[1,1].plot(ran['r'][locs]*lunit,ran['am_kep'][locs],':',label=r'$L_{\rm Kep}$',color='k')

    fig=ad.plot_phase(fig=fig,ax=axes[0,2],key=key,varname='r-vtot',weights=weights,xlabel=r'$r\rm\,[M]$',ylabel=r'$v\rm\,[km\,s^{-1}]$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,xunit=lunit,yunit=vunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[0,2].plot(ran['r'][locs]*lunit,ran['v_kep'][locs]*vunit,':',label=r'$v_{\rm Kep}$',color='k')

    fig=ad.plot_phase(fig=fig,ax=axes[0,3],key=key,varname='r-v_A',weights=weights,xlabel=r'$r\rm\,[M]$',ylabel=r'$v_A\rm\,[km\,s^{-1}]$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,xunit=lunit,yunit=vunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[0,3].plot(ran['r'][locs]*lunit,ran['v_kep'][locs]*vunit,':',label=r'$v_{\rm Kep}$',color='k')

    fig=ad.plot_phase(fig=fig,ax=axes[1,2],key=key,varname='r-btot',weights=weights,xlabel=r'$r\rm\,[M]$',ylabel=r'$|B|\rm\,[\mu G]$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,xunit=lunit,yunit=magnetic_unit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')

    fig=ad.plot_phase(fig=fig,ax=axes[1,3],key=key,varname='r-beta',weights=weights,xlabel=r'$r\rm\,[M]$',ylabel=r'$\beta$',label=r'$M\rm\,[M_\odot]$',
                      title=None,unit=munit,density=True,xunit=lunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')

    fig.suptitle(fr'Time={ad.time*tunit:.2f} Myr (Dashed lines: virial equilibrium)')
    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png")#,bbox_inches='tight'
    plt.close(fig)
    #'''

    #'''
    print('radial '+str(ad.num))
    #fig,axes=ak.subplots(2,2,figsize=(8,6),dpi=200,wspace=0.3,top=0.92,hspace=0.1,sharex=False,raw=True)
    fig,axes = ak.subplots(2,3,figsize=(12,6),dpi=128,top=0.92,wspace=0.25,left=0.05,right=0.95,raw=True)
    fig.suptitle(f'Time={ad.time*tunit:.2f} Myr')
    savelabel=f'radial_0'

    colors=['k','#3369E8','#009925','#FBBC05','#EA4335',]
    ylabels = [r'$n\rm\,[cm^{-3}]$',r'Time ratio',r'$\beta$',r'$T\rm\,[K]$',r'$\dot{M}\rm\,[M_\odot\,yr^{-1}]$',r'$L\rm\,[M]$',r'$p$']
    Tunit=unit.temperature_cgs
    locs=np.where((ran['r']-ad.rin)*(ran['r']-ad.profs['r_vol']['r'][-1])<=0)
    axes[0,0].plot(ran['r'][locs]*lunit,ran['dens'][locs],':',label=f'Initial',color=colors[0])
    axes[1,0].plot(ran['r'][locs]*lunit,ran['temp'][locs]*Tunit,':',color=colors[0])
    ad.profs['r_vol']['t_ff']=ak.asnumpy(xp.interp(xp.asarray(ad.profs['r_vol']['r']),xp.asarray(ran['r']),xp.asarray(ran['t_ff'])))
    #r2d=interp1d(ran['r'],ran['dens'])
    #axes[0,0].plot([4e-5,1e-3],[25*r2d(1e-3),r2d(1e-3)],'k--',label=r'$\propto r^{-1}$')

    #for i,suf,cn,label in zip([0,1,2],['_cold','_hot','',],[1,-1,0],['Cold','Hot','Total']):
    for i,suf,cn,label in zip([0,1,2,3],['_cold','_warm','_hot',''],[1,2,-1,0],['Cold','Warm','Hot','Total']):
        if ('r_vol'+suf not in ad.profs): continue
        r, rad, rad_v, rad_m = ad.profs['r_vol'+suf]['r'], ad.profs['r_vol'], ad.profs['r_vol'+suf], ad.profs['r_mass'+suf]
        axes[0,0].plot(r*lunit,rad_v['dens'],color=colors[cn],label=label)
        axes[1,0].plot(r*lunit,rad_m['temp']*Tunit,color=colors[cn],label=label)
        rad_m['tcool']=1/(ad.gamma-1)*rad_m['temp']/(rad_v['dens']*ak.CoolFnShure(rad_m['temp']*Tunit)/unit.cooling_cgs/acc.n_t_h_ratio**2)
        axes[0,1].plot(r*lunit,-r/rad_v['velin']/rad['t_ff'],marker='',color=colors[cn],label=(r'$t_{\rm inflow}/t_{\rm ff}$' if (i==3) else None))
        axes[0,1].plot(r*lunit,rad_m['tcool']/rad['t_ff'],linestyle='--',marker='',color=colors[cn],label=(r'$t_{\rm cool}/t_{\rm ff}$' if (i==3) else None))
        axes[0,2].plot(r*lunit,rad_v['eint']/rad_v['btot^2']*2,color=colors[cn],label=label)
        axes[1,2].plot(r*lunit,rad_v['amx'],color=colors[cn],label=label)
        axes[1,2].plot(r*lunit,rad_v['amy'],color=colors[cn],label=label,alpha=0.5)
        axes[1,2].plot(r*lunit,rad_v['amz'],color=colors[cn],label=label,alpha=0.3)
        for j,var,ls,sign,label in zip([0,1,2],['mdotin','mdotout','mdot'],['--',':','-'],[-1,1,-1],[r'$\dot{M}_{\rm inflow}$',r'$\dot{M}_{\rm outflow}$',r'$\dot{M}_{\rm net}$']):
            axes[1,1].plot(r*lunit,sign*rad[var+suf]*mdot_unit,linestyle=ls,color=colors[cn],label=label if i==3 else None)

    axes[0,0].set_ylim(5e-3)
    axes[0,1].set_ylim(1e-4,3e3)
    axes[1,0].set_ylim(1e5)
    axes[1,1].set_ylim(1e-4,3e2)
    axes[-1,0].set_xlabel(r'$r\rm\,[M]$')
    axes[-1,1].set_xlabel(r'$r\rm\,[M]$')
    axes[-1,-1].set_xlabel(r'$r\rm\,[M]$')
    for ax,ylabel in zip(axes.flat,ylabels):
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(ad.rmin*lunit,ad.rmax*lunit)
    for ax in axes[:-1,:].flat:
        ax.set_xticklabels([])
    axes[0,2].set_ylim(1.1e-3,9e4)
    axes[1,2].set_yscale('symlog')
    axes[1,2].set_ylim(-3e3,3e3)
    axes[0,0].legend(ncol=1,loc='best')
    axes[0,1].legend(ncol=1,loc='best')
    axes[1,1].legend(ncol=1,loc='best')
    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png")#,bbox_inches='tight'
    plt.close(fig)
    #'''

    return ad

def make_video(figlabel='',videolabel=None,figdir='',videodir=None,duration=0.05,fps=20):
    if (videodir is None): videodir=figdir
    if (videolabel is None): videolabel=figlabel
    from moviepy.editor import ImageClip, concatenate_videoclips
    numlist=[]
    for file in sorted(os.listdir(figdir)):
        if file.startswith(figlabel) and file.endswith('.png'):
            num=int((file.split('.')[-2]).split('_')[-1])
            numlist.append(num)
    numlist=sorted(list(set(numlist)))
    img = (f"{figdir}/{figlabel}_{i:04d}.png" for i in numlist)
    clips = [ImageClip(m).set_duration(duration) for m in img]
    concat_clip = concatenate_videoclips(clips, method="compose")
    if (not os.path.isdir(videodir)): os.mkdir(videodir)
    concat_clip.write_videofile(f"{videodir}/{videolabel}.mp4",fps=fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-p', '--path', type=str, default = '../simu')
    parser.add_argument('-d', '--data', type=str, default = 'data')
    parser.add_argument('-t', '--task', type=str, default = '')
    parser.add_argument('-a', '--all',  action='store_true', default=False)
    parser.add_argument('-b', '--beg', type=int, default = 0)
    parser.add_argument('-e', '--end', type=int, default = 0)
    parser.add_argument('--batch_size', type=int, default = 0)
    parser.add_argument('-v', '--variables', nargs='+', help='<Required> Set flag', default=['mhd_w_bcc','mhd_divb'])
    parser.add_argument('-z', '--zooms', nargs='+', help='<Required> Set flag', default=[])
    args = parser.parse_args()
    data_path=args.path+'/'+args.data+'/'
    task=args.task
    variables=args.variables
    zooms=args.zooms if (len(args.zooms)>0) else None
    print(variables)
    # get path
    binpath=data_path+'bin/'
    athdfpath=data_path+'athdf/'
    pklpath=data_path+'pkl/'
    figpath=data_path+'fig/'
    videopath=data_path+'video/'
    for path in [athdfpath,pklpath,figpath]:
        if not os.path.isdir(path):
            os.mkdir(path)
    # get numlist
    numlist=[]
    binfiles = [binpath+f for f in os.listdir(binpath) if f.endswith('.bin')]
    athdffiles = [athdfpath+f for f in os.listdir(athdfpath) if f.endswith('.athdf')]
    for file in sorted(binfiles+athdffiles):
        if file.endswith('.bin') or file.endswith('.athdf'):
            num=int(file.split('.')[-2])
            if (args.all or not os.path.isfile(pklpath+f'Base.{num:05d}.pkl') or
                 os.path.getmtime(file)>os.path.getmtime(pklpath+f'Base.{num:05d}.pkl')):
                numlist.append(num)
    numlist=sorted(list(set(numlist)))
    if (args.end-args.beg>0): numlist=list(range(args.beg,args.end))
    print('Work for', data_path, numlist)
    # run
    def run(i):
        for variable in variables:
            filename=athdfpath+f'Acc.{variable}.{i:05d}.athdf'
            binfilename=binpath+f'Acc.{variable}.{i:05d}.bin'
            if os.path.isfile(binfilename):
                if ((not os.path.isfile(filename)) \
                or (os.path.getmtime(binfilename)>os.path.getmtime(filename))):
                    ak.bin_to_athdf(binpath+f'Acc.{variable}.{i:05d}.bin',filename)
                    print(f'bin_to_athdf {filename}')
        ad=ak.AthenaData()
        #try:
        variable=variables[0]
        if True:
            print(f'running i={i}')
            if ('w' in task):
                print(f'loading athdf i={i}')
                filename=athdfpath+f'/Acc.{variable}.{i:05d}.athdf'
                ad.load(filename)
                print(f'working i={i}')
                adwork(ad,zooms).save(pklpath+f'/Base.{ad.num:05d}.pkl')
                #adwork(ad).save(ad.path.replace('athdf','h5')+f'/Base.{ad.num:05d}.h5')
            if ('p' in task):
                print(f'loading pkl i={i}')
                filename=pklpath+f'/Base.{i:05d}.pkl'
                ad.load(filename)
                ad.figpath=figpath
                print(f'plotting i={i}')
                adplot(ad,zooms)
            if ('u' in task):
                print(f'loading athdf i={i}')
                filename=athdfpath+f'/Acc.{variable}.{i:05d}.athdf'
                ad.load(filename)
                print(f'loading pkl i={i}')
                filename=pklpath+f'/Base.{i:05d}.pkl'
                ad.load(filename)
                print(f'updating i={i}')
                adupdate(ad).save(filename)
        #except Exception as excp:
        #    print(f"File {filename}:",sys.exc_info()[2].tb_frame,'\n',excp)
        del ad
        #xp._default_memory_pool.free_all_blocks()
        return

    tic=time.time()
    print("mp.cpu_count:",mp.cpu_count())
    for i in numlist:run(i)
    if ('v' in task):
        for label in ['phase_0','radial_0','image_x','image_y','image_z',]:
            make_video(figlabel='fig_'+label,videolabel='video_'+label,figdir=figpath,videodir=videopath,duration=0.05,fps=20)
    '''
    if args.batch_size>0:
        for i in range(0,len(numlist),args.batch_size):
            with mp.Pool(20) as p:p.map(run,numlist[i:i+args.batch_size])
            xp._default_memory_pool.free_all_blocks()
            gc.collect()
    else:
        with mp.Pool(20) as p:p.map(run,numlist)
    #'''
    toc=time.time()
    print(f"Time cost: {toc-tic:.2f}s")
    print("Done")
