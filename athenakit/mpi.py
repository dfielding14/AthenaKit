from . import macros
import numpy as np
if (macros.mpi_enabled):
    from mpi4py import MPI

    def Allreduce(arr,op=MPI.SUM,mpitype=None,**kwargs):
        # TODO(@mhguo): make sure it works for all types of arrays
        # comm = MPI.COMM_WORLD # gets communication pool 
        # if (mpitype is None):
        #     mpitype = MPI.DOUBLE
        # res = np.empty_like(arr)
        # comm.Barrier()
        # comm.Allreduce([arr, mpitype], [res, mpitype], op=MPI.SUM)
        res = np.empty_like(arr)
        MPI.COMM_WORLD.Barrier()
        MPI.COMM_WORLD.Allreduce(arr, res, op=op)
        return res

    def sum(arr,**kwargs):
        # comm = MPI.COMM_WORLD # gets communication pool 
        # if (mpitype is None):
        #     mpitype = MPI.DOUBLE
        # res = np.empty_like(arr)
        #comm.Barrier()
        #comm.Allreduce([arr, mpitype], [res, mpitype], op=MPI.SUM)
        # return res
        return Allreduce(arr,op=MPI.SUM,**kwargs)

    def max(arr,**kwargs):
        return Allreduce(arr,op=MPI.MAX,**kwargs)

    def min(arr,**kwargs):
        return Allreduce(arr,op=MPI.MIN,**kwargs)
