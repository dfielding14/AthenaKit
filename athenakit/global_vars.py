"""
Detect optional dependencies such as CuPy and MPI.
"""
### global variables ###

cupy_enabled = False
try:
    import cupy
    cupy.array(0)
    cupy_enabled = True
    print('CuPy enabled')
except:
    pass

mpi_enabled, rank, size = False, 0, 1
try:
    from mpi4py import MPI
    if (MPI.COMM_WORLD.Get_size() > 1):
        mpi_enabled = True
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        print('MPI enabled')
except:
    pass
