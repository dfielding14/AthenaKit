'''
Script for running analysis of athena++ simulation

Example:
    python -u py_job.py -p ../simulation -a -t wp
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

sys.path.append('/home/mg9443/Git')
import athenakit.athenakit as ak
from athenakit.athenakit.problem import acc

def adwork(ad,bins=256):
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

    varl=['dens','temp','velx','vely','bcc1','bcc2','pres','vtot',]
    jlist=[0,3,5,7,9]
    if(ad.mb_logical[:,-1].max()>11): jlist=[3,5,7,9,11]
    if(ad.mb_logical[:,-1].max()>14): jlist=[5,7,9,11,13]
    if(ad.mb_logical[:,-1].max()>17): jlist=[7,9,11,13,15]
    if(ad.mb_logical[:,-1].max()>19): jlist=[9,11,13,15,17]
    if(ad.mb_logical[:,-1].max()>21): jlist=[11,13,15,17,19]
    if(ad.mb_logical[:,-1].max()>23): jlist=[13,15,17,19,21]
    if(ad.mb_logical[:,-1].max()>25): jlist=[15,17,19,21,23]
    for j in jlist:
        #ad.set_slice(varl=varl,zoom=j,level=j+1)
        # TODO(@mhguo): use a real slice!
        # TODO(@mhguo): make less figures, possibly only dens, temp, vtot, 3*5 axes
        zoom=j
        ad.set_slice(['x','y'],varl=varl,key=f'z_{j}',weights='vol',bins=64,where=xp.abs(ad.data('z'))<ad.x3max/2**zoom/ad.nx3,
                     range=[[ad.x1min/2**zoom,ad.x1max/2**zoom],[ad.x2min/2**zoom,ad.x2max/2**zoom]])
    
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

    varl=['mdot','mdotin','mdotout','momdot','momdotin','momdotout','ekdot','ekdotin','ekdotout']
    for suf in ['','_cold','_warm','_hot']:
        ad.set_profile('r',varl=[var+suf for var in varl],key='r_vol',bins=bins,scales='log',weights='vol')

    varl=['dens','temp','velx','vely','velz','etot','ekin','eint','velr','pres','entropy',\
          'velin','velout','vtot','vrot','amx','amy','amz','amtot']
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
    
    varl=["dens",'temp','beta',"tran_velR","tran_velz","tran_velphi","tran_bccR","tran_bccz","tran_bccphi","tran_radial_flow",
          "tran_stress_zphi_reynolds","tran_stress_zphi_maxwell","tran_stress_zphi",
          "tran_stress_Rphi_reynolds","tran_stress_Rphi_maxwell","tran_stress_Rphi",
          ]
    Rmax=ad.rmax/2**7
    ad.set_profile2d(['tran_R','tran_z'],varl=varl,key='tRz_vol',bins=128,weights='vol',
                     where=np.logical_and(ad.data('tran_R')<Rmax,np.abs(ad.data('tran_z'))<Rmax))
    ad.set_profile('tran_R',varl=varl,key='tR_vol',bins=128,weights='vol',
                   where=(ad.data('tran_R')>ad.rin) & (ad.data('tran_R')<10*Rmax) & (np.abs(ad.data('tran_z/R'))<0.1))
    ad.set_profile('tran_z',varl=varl,key='tz_vol',bins=128,weights='vol',
                   where=(ad.data('tran_R')>0.1*Rmax) & (ad.data('tran_R')<Rmax) & (np.abs(ad.data('tran_z'))<Rmax))
    
    return ad

def adplot(ad):
    figdir=ad.path.replace('athdf','fig')+'/'
    unit=ad.unit
    ran=ad.rad_initial
    lunit,tunit,munit,vunit=unit.length_cgs/ak.pc_cgs,unit.time_cgs/ak.myr_cgs,unit.mass_cgs/ak.msun_cgs,unit.velocity_cgs/ak.km_s_cgs
    Tunit,mdot_unit,magnetic_unit=unit.temperature_cgs,unit.mdot_msun_yr,unit.magnetic_field_cgs/1e-6
    #'''
    #jlist=[0,3,5,7,9]
    jlist=[0,3,5,7,9]
    if(ad.mb_logical[:,-1].max()>11): jlist=[3,5,7,9,11]
    if(ad.mb_logical[:,-1].max()>14): jlist=[5,7,9,11,13]
    if(ad.mb_logical[:,-1].max()>17): jlist=[7,9,11,13,15]
    if(ad.mb_logical[:,-1].max()>19): jlist=[9,11,13,15,17]
    if(ad.rin<1e-3):  jlist=[9,11,13,15,17]
    if(ad.mb_logical[:,-1].max()>21): jlist=[11,13,15,17,19]
    if(ad.mb_logical[:,-1].max()>23): jlist=[13,15,17,19,21]
    if(ad.mb_logical[:,-1].max()>25): jlist=[15,17,19,21,23]
    fig,axes=ak.subplots(3,5,figsize=(20,10),dpi=128,wspace=0.5,raw=True)
    fig.suptitle(f'Time={ad.time*tunit}')
    for i,j in enumerate(jlist):#
        zoom=j
        vec=j-2
        xyz=None
        level=j+1
        xyunit=unit.length_cgs/ak.pc_cgs
        ad.plot_slice(fig=fig,ax=axes[0,i],var='dens',  key=f'z_{j}',save=False,label=r'$n\rm\,[cm^{-3}]$' if (j==jlist[-1]) else None,
            cmap='viridis',xlabel='X [pc]',ylabel='Y [pc]',zoom=zoom,vec=vec,level=level,
            xyz=xyz,circle=False,unit=1,                   xyunit=xyunit,dpi=128,norm='log',vmin=5e-3,vmax=1e4,
            title=f'Time = {ad.time:.2f}',aspect='equal')
        ad.plot_slice(fig=fig,ax=axes[1,i],var='temp',  key=f'z_{j}',save=False,label=r'$T\rm\,[K]$' if (j==jlist[-1]) else None,
            cmap='inferno',    xlabel='X [pc]',ylabel='',  zoom=zoom,vec=vec,level=level,
            xyz=xyz,circle=False,unit=unit.temperature_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=3e5,vmax=3e10,
            title=f'Time = {ad.time:.2f}',aspect='equal')
        ad.plot_slice(fig=fig,ax=axes[2,i],var='vtot',key=f'z_{j}',save=False,label=r'$|v|\rm\,[km\,s^{-1}]$' if (j==jlist[-1]) else None, 
            cmap='winter',xlabel='X [pc]',ylabel='Y [pc]',zoom=zoom,vec=vec,level=level,
            xyz=xyz,circle=False,unit=unit.velocity_cgs/ak.km_s_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e1,vmax=1e4,
            title=f'Time = {ad.time:.2f}',aspect='equal')
        #ad.plot_slice(fig=fig,ax=axes[1,1],var='pres',  key=f'z_{j}',save=False,label=r'$P\rm\,[k_B\,K\,cm^{-3}]$',
        #    cmap='plasma', xlabel='X [pc]',ylabel='',       zoom=zoom,vec=vec,level=level,
        #    xyz=xyz,circle=False,unit=unit.temperature_cgs,xyunit=xyunit,dpi=128,norm='log',vmin=1e5,vmax=1e13,
        #    title=f'Time = {ad.time:.2f}',aspect='equal')
        #ad.abins[i].plot_snapshot(N=2**5,varname='dens',save=True)
    savelabel=f'dtmp_all'
    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png",bbox_inches='tight')
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

    fig=ad.plot_phase(fig=fig,ax=axes[0,0],key=key,varname='dens_temp',weights=weights,xlabel=r'$n\rm\,[cm^{-3}]$',ylabel=r'$T\,\rm [K]$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,yunit=unit.temperature_cgs,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')

    fig=ad.plot_phase(fig=fig,ax=axes[0,1],key=key,varname='r_dens',weights=weights,xlabel=r'$r\rm\,[pc]$',ylabel=r'$n\,\rm [cm^{-3}]$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,xunit=lunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[0,1].plot(ran['r'][locs]*lunit,ran['dens'][locs],':',label=r'$n$',color='k')
    
    fig=ad.plot_phase(fig=fig,ax=axes[1,0],key=key,varname='r_temp',weights=weights,xlabel=r'$r\rm\,[pc]$',ylabel=r'$T\,\rm [K]$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,xunit=lunit,yunit=unit.temperature_cgs,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[1,0].plot(ran['r'][locs]*lunit,ran['temp'][locs]*Tunit,':',label=r'$T$',color='k')

    fig=ad.plot_phase(fig=fig,ax=axes[1,1],key=key,varname='r_amtot',weights=weights,xlabel=r'$r\rm\,[pc]$',ylabel=r'$L$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,xunit=lunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    axes[1,1].plot(ran['r'][locs]*lunit,ran['am_kep'][locs],':',label=r'$L_{\rm Kep}$',color='k')

    fig=ad.plot_phase(fig=fig,ax=axes[0,2],key=key,varname='r_vtot',weights=weights,xlabel=r'$r\rm\,[pc]$',ylabel=r'$v\rm\,[km\,s^{-1}]$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,xunit=lunit,yunit=vunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    fig=ad.plot_phase(fig=fig,ax=axes[0,3],key=key,varname='r_v_A',weights=weights,xlabel=r'$r\rm\,[pc]$',ylabel=r'$v_A\rm\,[km\,s^{-1}]$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,xunit=lunit,yunit=vunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    
    fig=ad.plot_phase(fig=fig,ax=axes[1,2],key=key,varname='r_btot',weights=weights,xlabel=r'$r\rm\,[pc]$',ylabel=r'$|B|\rm\,[\mu G]$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,xunit=lunit,yunit=magnetic_unit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    fig=ad.plot_phase(fig=fig,ax=axes[1,3],key=key,varname='r_beta',weights=weights,xlabel=r'$r\rm\,[pc]$',ylabel=r'$\beta$',label=r'$M\rm\,[M_\odot]$',
                      title='',unit=munit,density=True,xunit=lunit,norm='log',vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto')
    

    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png",bbox_inches='tight')
    plt.close(fig)
    #'''

    #'''
    print('radial '+str(ad.num))
    #fig,axes=ak.subplots(2,2,figsize=(8,6),dpi=200,wspace=0.3,top=0.92,hspace=0.1,sharex=False,raw=True)
    fig, axes = ak.subplots(2,2,top=0.92,wspace=0.05,figsize=(8,5.4),dpi=108)
    fig.suptitle(f'Time={ad.time*tunit} Myr')
    savelabel=f'radial_0'

    colors=['k','#3369E8','#009925','#FBBC05','#EA4335',]
    ylabels = [r'$n\rm\,[cm^{-3}]$',r'Time ratio',r'$T\rm\,[K]$',r'$\dot{M}\rm\,[M_\odot\,yr^{-1}]$',r'$p$',r'$P$']
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
        for j,var,ls,sign,label in zip([0,1,2],['mdotin','mdotout','mdot'],['--',':','-'],[-1,1,-1],[r'$\dot{M}_{\rm inflow}$',r'$\dot{M}_{\rm outflow}$',r'$\dot{M}_{\rm net}$']):
            axes[1,1].plot(r*lunit,sign*rad[var+suf]*mdot_unit,linestyle=ls,color=colors[cn],label=label if i==3 else None)

    axes[0,0].set_ylim(5e-3)
    axes[0,1].set_ylim(1e-1,3e3)
    axes[1,0].set_ylim(1e5)
    axes[1,1].set_ylim(1e-3,3e2)
    axes[-1,0].set_xlabel(r'$r\,[\rm pc]$')
    axes[-1,1].set_xlabel(r'$r\,[\rm pc]$')
    for n,ax in enumerate(axes.flat):
        ax.set_ylabel(ylabels[n])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.set_xlim(ad.rmin*lunit,ad.rmax*lunit)
    axes[0,0].legend(ncol=1,loc='best')
    axes[0,1].legend(ncol=1,loc='best')
    axes[1,1].legend(ncol=1,loc='best')
    fig.savefig(f"{figdir}/fig_{savelabel}_{ad.num:04d}.png",bbox_inches='tight')
    plt.close(fig)
    #'''

    return ad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-p', '--path', type=str, default = '../simu')
    parser.add_argument('-d', '--data', type=str, default = 'data')
    parser.add_argument('-t', '--task', type=str, default = '')
    parser.add_argument('-a', '--all',  action='store_true', default=False)
    args = parser.parse_args()
    data_path=args.path+'/'+args.data+'/'
    task=args.task
    # get path
    binpath=data_path+'bin/'
    athdfpath=data_path+'athdf/'
    pklpath=data_path+'pkl/'
    figpath=data_path+'fig/'
    for path in [athdfpath,pklpath,figpath]:
        if not os.path.isdir(path):
            os.mkdir(path)
    # get numlist
    numlist=[]
    for file in sorted(os.listdir(binpath)+os.listdir(athdfpath)):
        if file.endswith('.bin') or file.endswith('.athdf'):
            num=int(file.split('.')[-2])
            if (args.all or not os.path.isfile(pklpath+f'Base.{num:05d}.pkl')):
                numlist.append(num)
    numlist=sorted(list(set(numlist)))
    print('Work for', data_path, numlist)
    # run
    def run(i):
        filename=athdfpath+f'Acc.mhd_w_bcc.{i:05d}.athdf'
        if (args.all or not os.path.isfile(filename)) and os.path.isfile(binpath+f'Acc.mhd_w_bcc.{i:05d}.bin'):
            ak.bin_to_athdf(binpath+f'Acc.mhd_w_bcc.{i:05d}.bin',filename)
            ak.bin_to_athdf(binpath+f'Acc.mhd_divb.{i:05d}.bin',athdfpath+f'Acc.mhd_divb.{i:05d}.athdf')
        ad=ak.AthenaData()
        #try:
        if True:
            print(i)
            if ('w' in task):
                print(f'loading athdf i={i}...')
                filename=data_path+f'athdf/Acc.mhd_w_bcc.{i:05d}.athdf'
                ad.load(filename)
                print(f'working i={i}...')
                adwork(ad).save(ad.path.replace('athdf','pkl')+f'/Base.{ad.num:05d}.pkl')
                #adwork(ad).save(ad.path.replace('athdf','h5')+f'/Base.{ad.num:05d}.h5')
            if ('p' in task):
                print(f'loading pkl i={i}...')
                filename=data_path+f'pkl/Base.{i:05d}.pkl'
                ad.load(filename)
                print(f'plotting i={i}...')
                adplot(ad)
        #except Exception as excp:
        #    print(f"File {filename}:",sys.exc_info()[2].tb_frame,'\n',excp)
        return

    tic=time.time()
    #for i in numlist:run(i)
    with mp.Pool(16) as p:p.map(run,numlist)
    toc=time.time()
    print(f"Time cost: {toc-tic:.2f}s")
    print("Done")
