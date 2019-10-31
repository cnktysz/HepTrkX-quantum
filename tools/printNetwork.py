import numpy as np
import sys, os, time, datetime
sys.path.append(os.path.abspath(os.path.join('.')))
import matplotlib.pyplot as plt
from qiskit import *
from datasets.hitgraphs import get_datasets
import multiprocessing
from qnetworks.MERA import MERA_edge_forward

def map2angle(B):
	# Maps input features to 0-2PI
	r_min     = 0.
	r_max     = 1.
	phi_min   = -1.
	phi_max   = 1.
	z_min     = 0.
	z_max     = 1.2
	B[:,0] =  (B[:,0]-r_min)/(r_max-r_min) * 2 * np.pi 
	B[:,1] =  (B[:,1]-phi_min)/(phi_max-phi_min) * 2 * np.pi 
	B[:,2] =  (B[:,2]-z_min)/(z_max-z_min) * 2 * np.pi 
	B[:,3] =  (B[:,3]-r_min)/(r_max-r_min) * 2 * np.pi 
	B[:,4] =  (B[:,4]-phi_min)/(phi_max-phi_min) * 2 * np.pi 
	B[:,5] =  (B[:,5]-z_min)/(z_max-z_min) * 2 * np.pi 
	return B
def preprocess(data):
	X,Ro,Ri,y = data
	X[:,2] = np.abs(X[:,2]) # correction for negative z
	bo    = np.dot(Ro.T, X)
	bi    = np.dot(Ri.T, X)
	B     = np.concatenate((bo,bi),axis=1)
	return map2angle(B), y
############################################################################################
##### MAIN ######
if __name__ == '__main__':
	n_param = 19
	theta_learn = np.random.rand(n_param)*np.pi*2 
	input_dir   = 'data/hitgraphs_big' 
	n_files     = 2
	n_valid     = 1
	n_train     = 1
	train_data, valid_data = get_datasets(input_dir, n_train, n_valid)
	B, y = preprocess(train_data[0])
	MERA_edge_forward(B[0],theta_learn,draw=True)
	

	
