'''
Script for running analysis of athena++ simulation

Example:
    python -u script.py -p ../simulation -t wpv -v simu.hydro_w
    # for specific files
    python -u script.py -p ../simulation -t wpv -v simu.hydro_w --nlist 0 1 2
    # for specific range
    python -u script.py -p ../simulation -t wpv -v simu.hydro_w -b 0 -e 10 -s 2
    # for parallel run
    mpirun -n 2 python -u script.py -p ../simulation -t wpv -v simu.hydro_w
'''

import os
import sys
import argparse
import time
import numpy as np
try:
    import cupy as xp
    xp.array(0)
except:
    import numpy as xp
from matplotlib import pyplot as plt
import multiprocessing as mp
from pathlib import Path

sys.path.append('/home/mg9443/Git')
import athenakit.athenakit as ak
plt.style.use('~/Git/pykit/pykit/mplstyle/mg')

# TODO(@mhguo): seperate work and plot onto gpu and cpu respectively!

def adwork(ad,zlist=None,dlevel=1,bins=256):
    ad.rmin = ad.mb_dx.min()
    ad.rmax = float(np.min(np.abs([ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min,ad.x3max])))
    ad.data_raw.update(ak.physics.grmhd.variables(ad.data,ad.spin))
    ad.add_data_func('vkep', lambda d : d('1/r**0.5'))
    varl=['vol','mass']
    ad.set_sum(varl)
    ad.sums['bound']=ad.get_sum(varl,where=ad.data('Begas')<0)
    ad.sums['unbound']=ad.get_sum(varl,where=ad.data('Begas')>0)
    ad.sums['sph_rmax']=ad.get_sum(varl,where=ad.data('r')<0.75*ad.rmax)

    print('phase '+str(ad.num))
    varl=['r','dens','Begas','vphi/vkep']
    ad.set_hist(varl,weights='vol')
    ad.set_hist(varl,weights='mass')
    where=ad.data('Begas')>0
    ad.set_hist(varl,weights='vol',where=where,key='unbound')
    ad.set_hist(varl,weights='mass',where=where,key='unbound')
    where=ad.data('Begas')<0
    ad.set_hist(varl,weights='vol',where=where,key='bound')
    ad.set_hist(varl,weights='mass',where=where,key='bound')

    varl=[['r','dens'],['r','btot'],['r','pres']]
    ad.set_hist2d(varl,weights='vol',bins=bins,scales='log')
    varl=[['r','dens'],['r','temp'],['r','pres'],['r','amz']]
    ad.set_hist2d(varl,weights='mass',bins=bins,scales='log')

    print('profile '+str(ad.num))
    varl=['dens','temp','pres','wgas','dens*lor','dens*ur','wgas*u_t','wgas*ur','wgas*u_t*ur','wgas*u_ph','Tr_ph_hydro',
          'wmhd','wmhd*ur','wmhd*u_ph','Tr_ph_mhd','br','b_ph','b^2','br*b_ph']
    ad.set_profile2d(['r','theta'],varl=varl,key='rtheta_vol',bins=[bins,bins//4],weights='vol',scales=['log','linear'],
                    range=[[ad.rmin,ad.rmax],[0.0,np.pi]],)
    ad.set_profile('r',varl=varl,key='r_eqt',bins=bins,weights='vol',scales='log',where=ad.data('z^2')<ad.data('0.25*r^2'),
                    range=[[ad.rmin,ad.rmax],],)
    ad.set_profile('r',varl=varl,key='r_pol',bins=bins,weights='vol',scales='log',where=ad.data('z^2')>ad.data('0.75*r^2'),
                    range=[[ad.rmin,ad.rmax],],)
    return ad

def adupdate(ad,bins=256):
    ad.rmin = ad.mb_dx.min()
    ad.rmax = float(np.min(np.abs([ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min,ad.x3max])))
    ad.data_raw.update(ak.physics.grmhd.variables(ad.data,ad.spin))
    ad.add_data_func('vkep', lambda d : d('1/r**0.5'))
    varl=['dens','temp','vphi','vrot','ekin','pmag','uph']
    ad.set_profile('r',varl=varl,key='r_vol',bins=bins,scales='log',weights='vol',range=[[ad.rmin,ad.rmax]])
    ad.set_profile('r',varl=varl,key='r_mass',bins=bins,scales='log',weights='mass',range=[[ad.rmin,ad.rmax]])
    where=ad.data('Begas')>0
    ad.set_profile('r',varl=varl,key='r_vol_unbound',bins=bins,scales='log',weights='vol',where=where,range=[[ad.rmin,ad.rmax]])
    ad.set_profile('r',varl=varl,key='r_mass_unbound',bins=bins,scales='log',weights='mass',where=where,range=[[ad.rmin,ad.rmax]])
    where=ad.data('Begas')<0
    ad.set_profile('r',varl=varl,key='r_vol_bound',bins=bins,scales='log',weights='vol',where=where,range=[[ad.rmin,ad.rmax]])
    ad.set_profile('r',varl=varl,key='r_mass_bound',bins=bins,scales='log',weights='mass',where=where,range=[[ad.rmin,ad.rmax]])
    return ad

def adplot(ad,zlist=None,dlevel=1):
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
    tic=time.time()
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-p', '--path', type=str, default = '../simu')
    parser.add_argument('-d', '--data', type=str, default = 'data')
    parser.add_argument('-t', '--task', type=str, default = '')
    parser.add_argument('-a', '--all',  action='store_true', default=False)
    parser.add_argument('-b', '--beg', type=int, default = 0)
    parser.add_argument('-e', '--end', type=int, default = 0)
    parser.add_argument('-s', '--step', type=int, default = 1)
    parser.add_argument('--nlist', nargs='+', help='List of files', default=[])
    parser.add_argument('--batch_size', type=int, default = 0)
    parser.add_argument('-v', '--variables', nargs='+', help='<Required> Set flag', default=['mhd_w_bcc','mhd_divb'])
    parser.add_argument('-z', '--zooms', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('-l', '--level', type=int, default = 1)
    parser.add_argument('--bins', type=int, default = 256)
    parser.add_argument('-n', '--nprocess', type=int, default = 0)
    args = parser.parse_args()
    data_path=args.path+'/'+args.data+'/'
    task=args.task
    variables=args.variables
    zooms=args.zooms if (len(args.zooms)>0) else None
    nlist=args.nlist if (len(args.nlist)>0) else None
    dlevel=args.level
    bins=args.bins
    print(variables)
    # get path
    binpath=data_path+'bin/'
    athdfpath=data_path+'athdf/'
    pklpath=data_path+'pkl/'
    figpath=data_path+'fig/'
    videopath=data_path+'video/'
    if (ak.global_vars.rank==0):
        for path in [athdfpath,pklpath,figpath]:
            if not os.path.isdir(path):
                os.mkdir(path)
    if (ak.global_vars.mpi_enabled):
        ak.mpi.MPI.COMM_WORLD.Barrier()
    # get numlist
    numlist=[]
    binfiles = [binpath+f for f in os.listdir(binpath) if (f.startswith(variables[0]) and f.endswith('.bin'))]
    athdffiles = [athdfpath+f for f in os.listdir(athdfpath) if f.endswith('.athdf')]
    for file in sorted(binfiles+athdffiles):
        if file.endswith('.bin') or file.endswith('.athdf'):
            num=int(file.split('.')[-2])
            if (args.all or not os.path.isfile(pklpath+f'Base.{num:05d}.pkl') or
                 os.path.getmtime(file)>os.path.getmtime(pklpath+f'Base.{num:05d}.pkl')):
                numlist.append(num)
    numlist=sorted(list(set(numlist)))
    if (args.end-args.beg>0): numlist=list(range(args.beg,args.end,args.step))
    if (nlist is not None): numlist = [int(n) for n in nlist]
    print('Work for', data_path, numlist)
    # run
    def run(i):
        ad=ak.AthenaData()
        #try:
        variable=variables[0]
        if True:
            print(f'running i={i}')
            if ('c' in task):
                for variable in variables:
                    filename=athdfpath+f'{variable}.{i:05d}.athdf'
                    binfilename=binpath+f'{variable}.{i:05d}.bin'
                    if os.path.isfile(binfilename):
                        if ((not os.path.isfile(filename)) \
                        or (os.path.getmtime(binfilename)>os.path.getmtime(filename))):
                            ak.bin_to_athdf(binpath+f'{variable}.{i:05d}.bin',filename)
                            print(f'convert {binfilename} to {filename}')
            if ('w' in task):
                print(f'loading binary i={i}')
                # filename=athdfpath+f'{variable}.{i:05d}.athdf'
                filename=binpath+f'{variable}.{i:05d}.bin'
                ad.load(filename)
                print(f'working i={i}')
                adwork(ad,zooms,dlevel,bins).save(pklpath+f'/Base.{ad.num:05d}.pkl')
                #adwork(ad).save(ad.path.replace('athdf','h5')+f'/Base.{ad.num:05d}.h5')
            if ('u' in task):
                if ('w' not in task):
                    print(f'loading binary i={i}')
                    filename=binpath+f'/{variable}.{i:05d}.bin'
                    ad.load(filename)
                    print(f'loading pkl i={i}')
                    filename=pklpath+f'/Base.{i:05d}.pkl'
                    if os.path.isfile(filename):
                        ad.load(filename)
                print(f'updating i={i}')
                adupdate(ad,bins).save(pklpath+f'/Base.{ad.num:05d}.pkl')
            if ('p' in task and ak.global_vars.rank==0):
                print(f'loading pkl i={i}')
                filename=pklpath+f'/Base.{i:05d}.pkl'
                ad.load(filename)
                ad.figpath=figpath
                print(f'plotting i={i}')
                adplot(ad,zooms,dlevel)
        #except Exception as excp:
        #    print(f"File {filename}:",sys.exc_info()[2].tb_frame,'\n',excp)
        del ad
        #xp._default_memory_pool.free_all_blocks()
        return

    print("mp.cpu_count:",mp.cpu_count())
    if ('w' in task or 'u' in task or 'p' in task or 'c' in task):
        if args.nprocess<=1:
            for n in numlist:run(n)
        else:
            with mp.Pool(args.nprocess) as p:p.map(run,numlist)
    if ('f' in task and ak.global_vars.rank==0):
        numlist=[]
        for file in sorted(os.listdir(pklpath)):
            if file.endswith('.pkl'):
                num=int(file.split('.')[-2])
                figname=figpath+f'/fig_phase_0_{num:04d}.png'
                if (args.all or not os.path.isfile(figname) or 
                    os.path.getmtime(figname)<os.path.getmtime(pklpath+f'/Base.{num:05d}.pkl')):
                    numlist.append(num)
        numlist=sorted(list(set(numlist)))
        print('Plot for', data_path, numlist)
        def plot(i):
            ad=ak.AthenaData()
            ad.load(pklpath+f'/Base.{i:05d}.pkl')
            ad.figpath=figpath
            print(f'plotting i={i}')
            adplot(ad,zooms,dlevel)
        if (args.nprocess<=1):
            for i in numlist: plot(i)
        else:
            with mp.Pool(args.nprocess) as p:p.map(plot,numlist)
    if ('v' in task and ak.global_vars.rank==0):
        for label in ['image_x','image_y','image_z','phase_0','radial_0',]:
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
