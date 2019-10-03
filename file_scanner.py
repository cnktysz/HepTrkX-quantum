import numpy as np
import matplotlib.pyplot as plt
from datasets.hitgraphs import HitGraphDataset

def find_range(B):
	
	n_section = 8
	r_min     = 0
	r_max     = 1.0
	phi_min   = -1
	phi_max   = 1
	z_min     = 0
	z_max     = 1.2
	B[:,0] = 2*np.pi * (B[:,0]-r_min)/(r_max-r_min) 
	B[:,1] = 2*np.pi * (B[:,1]-phi_min)/(phi_max-phi_min) 
	B[:,2] = 2*np.pi * (B[:,2]-z_min)/(z_max-z_min) 
	B[:,3] = 2*np.pi * (B[:,3]-r_min)/(r_max-r_min) 
	B[:,4] = 2*np.pi * (B[:,4]-phi_min)/(phi_max-phi_min) 
	B[:,5] = 2*np.pi * (B[:,5]-z_min)/(z_max-z_min)
	
	r_min_o   = min(B[:,0])  
	r_min_i   = min(B[:,3])  
	phi_min_o = min(B[:,1])  
	phi_min_i = min(B[:,4])  
	z_min_o   = min(B[:,2])  
	z_min_i   = min(B[:,5])  
	r_max_o   = min(B[:,0])  
	r_max_i   = max(B[:,3])  
	phi_max_o = max(B[:,1])  
	phi_max_i = max(B[:,4])  
	z_max_o   = max(B[:,2])  
	z_max_i   = max(B[:,5]) 
	r_min 	= min(r_min_o,r_min_i)
	r_max   = max(r_max_o,r_max_i)
	phi_min = min(phi_min_o,phi_min_i)
	phi_max = max(phi_max_o,phi_max_i)
	z_min 	= min(z_min_o,z_min_i)
	z_max 	= max(z_max_o,z_max_i)
	return [r_min,r_max,phi_min,phi_max,z_min,z_max]


if __name__ == '__main__':
	input_dir = 'data\hitgraphs_big'
	n_files = 16*100
	data = HitGraphDataset(input_dir, n_files)
	for n_file in range(n_files):
		X,Ro,Ri,y = data[n_file]
		if n_file%2==0:
			X[:,2] = -X[:,2]
		bo     = np.dot(Ro.T, X)
		bi     = np.dot(Ri.T, X)
		B      = np.concatenate((bo,bi),axis=1)
		ranges = find_range(B)
		print(str(n_file) + ' --> ' + str(ranges))