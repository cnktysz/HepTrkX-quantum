import numpy as np
import sys
from datasets.hitgraphs import HitGraphDataset
import dask.array as da 
import time
from dask.distributed import Client, progress
from multiprocessing import cpu_count

client = Client(processes=False, threads_per_worker=1,
                n_workers=8, memory_limit='2GB')

client

def map2angle(B):
	# Maps input features to 0-2PI
	C = np.zeros((len(B),len(B[0])))
	n_section = 8
	r_min 	  = 0
	r_max     = 1.200
	phi_min   = -np.pi/n_section
	phi_max   = np.pi/n_section
	z_min 	  = -1.200
	z_max 	  = 1.200
	C[:,0] = 2*np.pi * (B[:,0]-r_min)/(r_max-r_min) 
	C[:,1] = 2*np.pi * (B[:,1]-phi_min)/(phi_max-phi_min) 
	C[:,2] = 2*np.pi * (B[:,2]-z_min)/(z_max-z_min) 
	C[:,3] = 2*np.pi * (B[:,3]-r_min)/(r_max-r_min) 
	C[:,4] = 2*np.pi * (B[:,4]-phi_min)/(phi_max-phi_min) 
	C[:,5] = 2*np.pi * (B[:,5]-z_min)/(z_max-z_min)
	return C

input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs'
data = HitGraphDataset(input_dir, 1)
X,Ro,Ri,y = data[0]
bo   = np.dot(Ro.T, X)
bi   = np.dot(Ri.T, X)
B    = np.concatenate((bo,bi),axis=1)

arr = np.random.rand(10000,10000)

darr = da.from_array(arr,chunks=(250,250))
print(darr.npartitions)

time0 = time.time()
for i in range(20):
	arr.sum()
time1 = time.time()
print('Numpy array: ' + str(time1-time0))

time2 = time.time()
for i in range(2000):
	darr.sum()
time3 = time.time()
print('Dask array: ' + str(time3-time2))


