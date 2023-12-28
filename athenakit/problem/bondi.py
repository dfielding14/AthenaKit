import numpy as np
try:
    import cupy as xp
    xp.array(0)
except:
    import numpy as xp
from numpy.linalg import inv
from .. import physics
from .. import units
from .. import kit
from ..athena_data import asnumpy

def add_tools(ad):
    ad.rmin = float(asnumpy(np.min(ad.data('r').min())))
    ad.rmax = float(np.min(np.abs([ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min,ad.x3max])))
            
    return

def add_data(ad,is_gr=False,add_bcc=True):
    if ('bcc1' not in ad.data_raw.keys()) and add_bcc:
        ad.add_data_func('bcc1', lambda sf : sf.data('zeros'))
        ad.add_data_func('bcc2', lambda sf : sf.data('zeros'))
        ad.add_data_func('bcc3', lambda sf : sf.data('zeros'))

    for var in ['mdot','mdotin','mdotout','momdot','momdotin','momdotout','ekdot','ekdotin','ekdotout']:
        ad.add_data_func(var, lambda sf, var=var : 4.0*xp.pi*sf.data('r')**2*sf.data(var.replace('dot','flxr')))

    if (is_gr): add_gr_data(ad)

    return

# TODO(@mhguo): not finished
def add_gr_data(ad):
    # Set metric
    a = ad.spin = ad.header("coord","a",float,0.0)
    ad.add_data_func('rks', lambda sf : physics.kerr_schild_radius(sf.data('x'),sf.data('y'),sf.data('z'),a))
    
    glower, gupper=physics.kerr_schild_metric_and_inverse(ad.data('x'),ad.data('y'),ad.data('z'),a)
    vx = ad.data('velx')
    vy = ad.data('vely')
    vz = ad.data('velz')
    bx, by, bz = ad.data('bccx'), ad.data('bccy'), ad.data('bccz')
    q = glower[1][1]*vx*vx + 2.0*glower[1][2]*vx*vy + \
        2.0*glower[1][3]*vx*vz + glower[2][2]*vy*vy + \
        2.0*glower[2][3]*vy*vz + glower[3][3]*vz*vz
    alpha = (-1.0/gupper[0][0])**0.5
    lor = (1.0 + q)**0.5
    u0 = lor/alpha
    u1 = vx - alpha * lor * gupper[0][1]
    u2 = vy - alpha * lor * gupper[0][2]
    u3 = vz - alpha * lor * gupper[0][3]
    u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3
    u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3
    u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3
    u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3
    b0 = u_1*bx + u_2*by + u_3*bz
    b1 = (bx + b0 * u1) / u0
    b2 = (by + b0 * u2) / u0
    b3 = (bz + b0 * u3) / u0
    b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3
    b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3
    b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3
    b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3
    b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3
    a2 = a**2
    r2 = ad.data('r')**2
    rks = ad.data('rks')
    rks2 = ad.data('rks')**2
    sth = xp.sin(ad.data('theta'))
    sph = xp.sin(ad.data('phi'))
    cph = xp.cos(ad.data('phi'))
    drdx = rks*ad.data('x')/(2.0*rks2 - r2 + a2)
    drdy = rks*ad.data('y')/(2.0*rks2 - r2 + a2)
    drdz = (rks*ad.data('z')+a2*ad.data('z')/rks)/(2.0*rks2 - r2 + a2)
    ur = drdx *u1 + drdy *u2 + drdz *u3
    br = drdx *b1 + drdy *b2 + drdz *b3
    u_ph = (-rks*sph-ad.spin*cph)*sth*u_1 + (rks*cph-ad.spin*sph)*sth*u_2
    b_ph = (-rks*sph-ad.spin*cph)*sth*b_1 + (rks*cph-ad.spin*sph)*sth*b_2
    
    ad.add_data('alpha',alpha)
    ad.add_data('u0',u0)
    ad.add_data('u1',u1)
    ad.add_data('u2',u2)
    #ad.add_data_func('ux', lambda sf : sf.data('velx')/sf.data('mdot'))
    #ad.add_data_func('ur', lambda sf : sf.data('velr'))
