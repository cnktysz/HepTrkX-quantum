import numpy as np
import sys, os, time, datetime
sys.path.append(os.path.abspath(os.path.join('.')))
import matplotlib.pyplot as plt
from qiskit import *
from datasets.hitgraphs import get_datasets
import multiprocessing
from qnetworks.TTN8 import TTN_edge_forward

############################################################################################
##### MAIN ######
if __name__ == '__main__':
	n_param = 15
	theta_learn = np.random.rand(n_param)*np.pi*2 
	edge = np.random.rand(8) * np.pi * 2
	TTN_edge_forward(edge,theta_learn,draw=True)
	

	
