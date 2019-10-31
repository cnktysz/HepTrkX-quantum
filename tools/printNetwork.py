import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from datasets.hitgraphs import get_datasets
import sys, time, datetime
import multiprocessing

def TTN_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	q       = QuantumRegister(len(edge))
	c       = ClassicalRegister(1)
	circuit = QuantumCircuit(q,c)
	# STATE PREPARATION
	for i in range(len(edge)):
		circuit.ry(edge[i],q[i])
	# APPLY forward sequence
	circuit.ry(theta_learn[0],q[0])
	circuit.ry(theta_learn[1],q[1])
	circuit.cx(q[0],q[1])
	circuit.ry(theta_learn[2],q[2])
	circuit.ry(theta_learn[3],q[3])
	circuit.cx(q[2],q[3])
	circuit.ry(theta_learn[4],q[4])
	circuit.ry(theta_learn[5],q[5])
	circuit.cx(q[5],q[4]) # reverse the order
	circuit.ry(theta_learn[6],q[1])
	circuit.ry(theta_learn[7],q[3])
	circuit.cx(q[1],q[3])
	circuit.ry(theta_learn[8],q[3])
	circuit.ry(theta_learn[9],q[4])
	circuit.cx(q[3],q[4])
	circuit.ry(theta_learn[10],q[4])
	# Qasm Backend
	circuit.measure(q[4],c)
	circuit.draw(filename='png/circuit/TTNcircuit.pdf')
	backend = Aer.get_backend('qasm_simulator')
	result = execute(circuit, backend, shots=1000).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/1000
	return(out)

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
	n_param = 11
	theta_learn = np.random.rand(n_param)*np.pi*2 
	input_dir   = 'data/hitgraphs_big' 
	n_files     = 2
	n_valid     = 1
	n_train     = 1
	train_data, valid_data = get_datasets(input_dir, n_train, n_valid)
	B, y = preprocess(train_data[0])
	TTN_edge_forward(B[0],theta_learn)
	

	
