import numpy as np
try:
    import cupy as xp
    xp.array(0)
except:
    import numpy as xp

def kerr_schild_radius(x,y,z,a):
    rad = (x**2+y**2+z**2)**0.5
    r = ((rad**2-a**2+((rad**2-a**2)**2+4.0*a**2*z**2)**0.5)/2.0)**0.5
    return r

def kerr_schild_metric_and_inverse(x,y,z,a):
    x = xp.asarray(x)
    y = xp.asarray(y)
    z = xp.asarray(z)
    rad = (x**2+y**2+z**2)**0.5
    # Kerr-Schild radius
    r = ((rad**2-a**2+((rad**2-a**2)**2+4.0*a**2*z**2)**0.5)/2.0)**0.5
    
    # Set covariant components
    # null vector l
    l_lower = xp.zeros((4,)+x.shape)
    l_lower[0] = xp.ones(x.shape)
    l_lower[1] = (r*x + (a)*y)/( (r)**2 + (a)**2 )
    l_lower[2] = (r*y - (a)*x)/( (r)**2 + (a)**2 )
    l_lower[3] = z/r

    # g_nm = f*l_n*l_m + eta_nm, where eta_nm is Minkowski metric
    f = 2.0 * (r)**2*r / (((r)**2)**2 + (a)**2*(z)**2)
    glower = xp.zeros((4,4)+x.shape)
    glower[0][0] = f * l_lower[0]*l_lower[0] - 1.0
    glower[0][1] = f * l_lower[0]*l_lower[1]
    glower[0][2] = f * l_lower[0]*l_lower[2]
    glower[0][3] = f * l_lower[0]*l_lower[3]
    glower[1][0] = glower[0][1]
    glower[1][1] = f * l_lower[1]*l_lower[1] + 1.0
    glower[1][2] = f * l_lower[1]*l_lower[2]
    glower[1][3] = f * l_lower[1]*l_lower[3]
    glower[2][0] = glower[0][2]
    glower[2][1] = glower[1][2]
    glower[2][2] = f * l_lower[2]*l_lower[2] + 1.0
    glower[2][3] = f * l_lower[2]*l_lower[3]
    glower[3][0] = glower[0][3]
    glower[3][1] = glower[1][3]
    glower[3][2] = glower[2][3]
    glower[3][3] = f * l_lower[3]*l_lower[3] + 1.0

    # Set contravariant components
    # null vector l
    l_upper = xp.zeros((4,)+x.shape)
    l_upper[0] = -1.0
    l_upper[1] = l_lower[1]
    l_upper[2] = l_lower[2]
    l_upper[3] = l_lower[3]

    # g^nm = -f*l^n*l^m + eta^nm, where eta^nm is Minkowski metric
    gupper = xp.zeros((4,4)+x.shape)
    gupper[0][0] = -f * l_upper[0]*l_upper[0] - 1.0
    gupper[0][1] = -f * l_upper[0]*l_upper[1]
    gupper[0][2] = -f * l_upper[0]*l_upper[2]
    gupper[0][3] = -f * l_upper[0]*l_upper[3]
    gupper[1][0] = gupper[0][1]
    gupper[1][1] = -f * l_upper[1]*l_upper[1] + 1.0
    gupper[1][2] = -f * l_upper[1]*l_upper[2]
    gupper[1][3] = -f * l_upper[1]*l_upper[3]
    gupper[2][0] = gupper[0][2]
    gupper[2][1] = gupper[1][2]
    gupper[2][2] = -f * l_upper[2]*l_upper[2] + 1.0
    gupper[2][3] = -f * l_upper[2]*l_upper[3]
    gupper[3][0] = gupper[0][3]
    gupper[3][1] = gupper[1][3]
    gupper[3][2] = gupper[2][3]
    gupper[3][3] = -f * l_upper[3]*l_upper[3] + 1.0

    return glower, gupper
