import numpy as np
import sys
import dask.array as da 
import time
from dask.distributed import Client, progress
from multiprocessing import cpu_count

client = Client(processes=False, threads_per_worker=4,
                n_workers=25, memory_limit='2GB')
client

x = np.random.rand(10000,10000)
darr = da.from_array(x,chunks=(1000,1000))
print(darr.npartitions)

time0 = time.time()
b = x.sum()
time1 = time.time()
print('Numpy only: ' + str(time1-time0))


time2 = time.time()
c = darr.sum()
time3 = time.time()
print('Dask array: ' + str(time3-time2))