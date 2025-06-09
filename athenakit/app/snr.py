import numpy as np
import yt
import pyxsim

def to_yt(data,shape,bbox,length_unit="pc",nprocs=1,default_species_fields='ionized',periodicity=(False,False,False)):
    ds = yt.load_uniform_grid(data, shape, length_unit=length_unit, bbox=bbox, nprocs=nprocs,
                              default_species_fields=default_species_fields,periodicity=periodicity)
    return ds

def xray(ds):
    #source_model = pyxsim.CIESourceModel("spex", 0.05, 11.0, 10, 0.3, binscale="log")
    source_model = pyxsim.CIESourceModel("spex", 0.5, 7.0, 10, 1.0, binscale="log")
    xray_fields = source_model.make_source_fields(ds, 0.5, 7.0)
    ad=ds.all_data()
    xray_emissivity = ad[("gas", "xray_emissivity_0.5_7.0_keV")]
    return xray_emissivity
