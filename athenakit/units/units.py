###############################################################
###            some basic parameters in cgs unit            ###
###############################################################

pi = CONST_PI = 3.141592653589793

###############################################################

cm_cgs = 1.0;                           # cm
g_cgs = 1.0;                            # g
s_cgs = 1.0;                            # s
cm_s_cgs = 1.0;                         # cm/s
g_cm3_cgs = 1.0;                        # g/cm^3
erg_cgs = 1.0;                          # erg
dyne_cm2_cgs = 1.0;                     # dyne/cm^2
kelvin_cgs = 1.0;                       # k

###############################################################

km_cgs = 1e5                                                  # cm
kms_cgs = km_s_cgs = 1.0e5                                    # cm/s
AU_cgs = 1.496e13                                             # cm
ly_cgs = 9.460471451897088e17                                 # cm
pc_cgs = parsec_cgs = 3.0856775809623245e+18                  # cm
kpc_cgs = 3.0856775809623245e+21                              # cm
Mpc_cgs = 3.0856775809623245e+24                              # cm
yr_cgs = CONST_YR_cgs = 3.15576e+7                            # s
kyr_cgs = CONST_KYR_cgs = 3.15576e+10                         # s
myr_cgs = Myr_cgs = CONST_MYR_cgs = 3.15576e+13               # s
msun_cgs = Msun_cgs = CONST_SOLAR_MASS_cgs = 1.98841586e+33   # g
rsun_cgs = Rsun_cgs = CONST_SOLAR_RADIUS_cgs = 6.957e10       # cm
Lsun_cgs = CONST_SOLAR_LUMINOSITY_cgs = 3.846e33              # erg/s
Tsun_cgs = CONST_SOLAR_TEMPERATURE_cgs = 5.780e3              # k
mearth_cgs = Mearth_cgs = CONST_EARTH_MASS_cgs = 5.965e27     # g
rearth_cgs = Rearth_cgs = CONST_EARTH_RADIUS_cgs = 6.371e8    # cm

###############################################################

c_cgs = speed_of_light_cgs = C_cgs = CONST_C_cgs = 2.99792458e10               # cm/s
G_cgs = grav_constant_cgs = CONST_G_cgs = 6.67408e-8                           # cm^3/(g*s^2)
e_cgs = CONST_E_cgs = CONST_ELECTRON_CHARGE_cgs = 4.8032e-10                   # cm^1.5*g^0.5/s
h_cgs = H_PLANCK_cgs = CONST_H_PLANCK_cgs = 6.6260755e-27                      # g*cm^2/s
hbar_cgs = HBAR_PLANCK_cgs = CONST_HBAR_PLANCK_cgs = 1.05457266e-27            # g*cm^2/s

###############################################################

mu_cgs = atomic_mass_unit_cgs = CONST_ATOM_MASS_UNIT_cgs = 1.660538921e-24     # g
mp_cgs = CONST_PROTON_MASS_cgs = 1.6726e-24                                    # g
mn_cgs = CONST_NEUTRON_MASS_cgs = 1.6749286e-24                                # g
me_cgs = CONST_ELECTRON_MASS_cgs = 9.1094e-28                                  # g
mH_cgs = CONST_HYDROGEN_MASS_cgs = 1.6733e-24                                  # g
kB_cgs = k_boltzmann_cgs = CONST_K_BOLTZMANN_cgs = 1.3806488e-16               # erg/k
eV_cgs = CONST_ELECTRON_VOLT_cgs = 1.6021772e-12                               # erg
keV_cgs = 1.6021772e-9                                                         # erg
sigmaSB_cgs = CONST_SIGMA_STEFAN_BOLTZMANN_cgs = 5.6705e-5                     # erg/s/cm^2/k^4
sigmaT_cgs = CONST_THOMSON_CROSS_SECTION_cgs = 6.6525e-25                      # cm^2
NA_cgs = CONST_AVAGADRO_NUMBER_cgs = 6.022140857747e23
alphaf_cgs = CONST_ALPHA_FINE_STRUCTURE_cgs = 7.2974e-3

###############################################################

arcsecond = 4.84813681109536e-06
Jy_cgs = Jansky_cgs = 1e-23
LEdd_cgs = 1.263e38                      # erg/s
MEdd_cgs = 1.405277e+18                  # g/s
kappaes_cgs = mp_cgs/sigmaT_cgs
mui = 1.23
mue = 1.14

##############################################################

# Units

class Units:
    # default unit is cgs
    def __init__(self,lunit=1.0,munit=1.0,tunit=1.0,mu=1.0):
        self.length_cgs=lunit
        self.mass_cgs=munit
        self.time_cgs=tunit
        self.mu=mu
        # set all the constants as attributes
        for name, value in globals().items():
            # check if the variable is a constant
            if ((not name.startswith("_")) and isinstance(value, (int, float))):
                setattr(self, name, value)
                # setattr(self, f"_{name}", value)  # Set it as a private attribute
                # Define a getter function for each attribute
                # getter = lambda self, name=name: getattr(self, f"_{name}")
                # Set it as a property on the class
                # setattr(self, name, property(getter))

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
        return self.velocity_cgs**2*self.mu*self.atomic_mass_unit_cgs/self.k_boltzmann_cgs
    @property
    def grav_constant(self):
        return self.grav_constant_cgs*self.density_cgs*self.time_cgs**2
    @property
    def speed_of_light(self):
        return self.speed_of_light_cgs/self.velocity_cgs
    @property
    def number_density_cgs(self):
        return self.density_cgs/self.mu/self.atomic_mass_unit_cgs
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
    def entropy_kevcm2(self,gamma=5./3.):
        return self.pressure_cgs/self.keV_cgs/self.number_density_cgs**gamma
    @property
    def magnetic_field_cgs(self):
        return (4.0*self.pi*self.density_cgs)**0.5*self.velocity_cgs

# unit=Units(lunit=kpc_cgs,munit=0.618*atomic_mass_unit_cgs*kpc_cgs**3,mu=0.618)
unit=Units()
