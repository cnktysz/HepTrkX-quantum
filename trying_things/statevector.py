
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.extensions.simulator import snapshot
from datasets.hitgraphs import HitGraphDataset
import sys, time
import multiprocessing

def TTN_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output

	backend = Aer.get_backend('statevector_simulator')
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


	result = execute(circuit, backend).result()
	statevector = result.get_statevector(circuit)
	out = sum(statevector[16:32]*statevector[16:32]) + sum(statevector[48:]*statevector[48:])
	print(out.real)

def map2angle(B):
	# Maps input features to 0-2PI
	r_min     = 0.
	r_max     = 1.
	phi_min   = -1.
	phi_max   = 1.
	z_min     = 0.
	z_max     = 1.2
	B[:,0] =  (B[:,0]-r_min)/(r_max-r_min) 
	B[:,1] =  (B[:,1]-phi_min)/(phi_max-phi_min) 
	B[:,2] =  (B[:,2]-z_min)/(z_max-z_min) 
	B[:,3] =  (B[:,3]-r_min)/(r_max-r_min) 
	B[:,4] =  (B[:,4]-phi_min)/(phi_max-phi_min) 
	B[:,5] =  (B[:,5]-z_min)/(z_max-z_min)
	return B
############################################################################################
if __name__ == '__main__':
	theta_learn = np.random.rand(11)*np.pi*2
	input_dir = 'data\hitgraphs_big'
	n_files = 1
	data = HitGraphDataset(input_dir, n_files)
	X,Ro,Ri,y = data[0]
	bo    = np.dot(Ro.T, X)
	bi    = np.dot(Ri.T, X)
	B     = np.concatenate((bo,bi),axis=1)
	B     = map2angle(B)

	TTN_edge_forward(B[0],theta_learn)