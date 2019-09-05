# Author: Cenk Tüysüz
# Date: 29.08.2019
# First attempt to test QuantumEdgeNetwork
# Run this code the train and test the network

import numpy as np
import matplotlib.pyplot as plt
from qiskit import(
	QuantumCircuit,
	QuantumRegister,
	ClassicalRegister,
	execute,
	BasicAer)
from qiskit.aqua.operator import Operator
from qiskit.aqua.components.initial_states import Zero
from qiskit.visualization import plot_histogram

from datasets.hitgraphs import HitGraphDataset
import sys


def TTN_edge_forward(B,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	backend = BasicAer.get_backend('qasm_simulator')
	q       = QuantumRegister(len(B))
	c       = ClassicalRegister(1)
	circuit = QuantumCircuit(q,c)
	# STATE PREPARATION
	for i in range(len(B)):
		circuit.ry(B[i],q[i])
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
	
	circuit.measure(q[4],c)

	#circuit.draw(filename='circuit.png')
	result = execute(circuit, backend, shots=100).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/100
	return(out)

def TTN_edge_back(input_,theta_learn,lr,error,label):
	# This function calculates the gradients for all learning 
	# variables numerically and updates them accordingly.
	# TODO: need to choose epislon properly
	epsilon = 0.05 # to take derivative
	gradient = np.zeros(len(theta_learn))
	update = np.zeros(len(theta_learn))
	for i in range(len(theta_learn)):
		## Compute f(x+epsilon)
		theta_learn[i] = (theta_learn[i] + epsilon)%(2*np.pi)
		## Evaluate
		out_plus = TTN_edge_forward(input_,theta_learn)
		## Compute f(x-epsilon)
		theta_learn[i] = (theta_learn[i] - 2*epsilon)%(2*np.pi)
		## Evaluate
		out_minus = TTN_edge_forward(input_,theta_learn)
		# Compute the gradient numerically
		gradient[i] = (out_plus-out_minus)/(2*epsilon)
		## Bring theta to its original value
		theta_learn[i] = (theta_learn[i] + epsilon)%(2*np.pi)
	
	## UPDATE theta_learn
	#print('gradient:' + str(gradient))
	update = lr*2*error*gradient
	#print('update:' + str(update))
	theta_learn = (theta_learn - update)%(2*np.pi)
	return theta_learn
#######################################################
def normalize(B):
	# This function takes the input matrix and maps it linearly
	# to 0-2PI.
	# TODO: Instead of this method use physical max and min to map.
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
	r_min 	  = min(r_min_o,r_min_i)
	r_max     = max(r_max_o,r_max_i)
	phi_min   = min(phi_min_o,phi_min_i)
	phi_max   = max(phi_max_o,phi_max_i)
	z_min 	  = min(z_min_o,z_min_i)
	z_max 	  = max(z_max_o,z_max_i)
	#print('r: '   + str(r_min)   + ' - ' + str(r_max))
	#print('phi: ' + str(phi_min) + ' - ' + str(phi_max))
	#print('z: '   + str(z_min)   + ' - ' + str(z_max))
	# Map between 0 - 2PI
	B[:,0] = 2*np.pi * (B[:,0]-r_min)/(r_max-r_min) 
	B[:,1] = 2*np.pi * (B[:,1]-phi_min)/(phi_max-phi_min) 
	B[:,2] = 2*np.pi * (B[:,2]-z_min)/(z_max-z_min) 
	B[:,3] = 2*np.pi * (B[:,3]-r_min)/(r_max-r_min) 
	B[:,4] = 2*np.pi * (B[:,4]-phi_min)/(phi_max-phi_min) 
	B[:,5] = 2*np.pi * (B[:,5]-z_min)/(z_max-z_min)
	return B 
def map2angle(B):
	# Maps input features to 0-2PI
	n_section = 4
	r_min 	  = 0
	r_max     = 1200
	phi_min   = -np.pi/n_section
	phi_max   = np.pi/n_section
	z_min 	  = -1.200
	z_max 	  = 1.200
	B[:,0] = 2*np.pi * (B[:,0]-r_min)/(r_max-r_min) 
	B[:,1] = 2*np.pi * (B[:,1]-phi_min)/(phi_max-phi_min) 
	B[:,2] = 2*np.pi * (B[:,2]-z_min)/(z_max-z_min) 
	B[:,3] = 2*np.pi * (B[:,3]-r_min)/(r_max-r_min) 
	B[:,4] = 2*np.pi * (B[:,4]-phi_min)/(phi_max-phi_min) 
	B[:,5] = 2*np.pi * (B[:,5]-z_min)/(z_max-z_min)
	return B

def test_accuracy(B,theta_learn,y):
	# This function only test the accuracy over a very limited set of data
	# due to time constraints
	# TODO: Need to test properly
	#input_dir = '/home/cenktuysuz/MyRepos/HepTrkX-quantum/data/hitgraphs'
	input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs'
	data = HitGraphDataset(input_dir, 3)
	X,Ro,Ri,y = data[2]
	bo   = np.dot(Ro.T, X)
	bi   = np.dot(Ri.T, X)
	B    = np.concatenate((bo,bi),axis=1)
	B    = normalize(B)
	acc  = 0
	size = 50
	for i in range(size):
		out = TTN_edge_forward(B[i],theta_learn)
		#print(str(i) + ': Result: ' + str(out) + ' Expected: ' + str(y[i]))
		if(y[i]==0):
			acc = acc + 1 - out
		else:
			acc = acc + out
	acc = 100.0 * acc/size
	print('Total Accuracy: ' + str(acc) + ' %')
	print('Theta_learn: ' + str(theta_learn))
	return acc
############################################################################################
##### MAIN ######
theta_learn = np.random.rand(11)*np.pi*2
lr = 0.01
EVERY_N_epoch = 500

#input_dir = '/home/cenktuysuz/MyRepos/HepTrkX-quantum/data/hitgraphs'
input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs'
n_files = 2
for n_file in range(n_files):
	data = HitGraphDataset(input_dir, n_files)
	X,Ro,Ri,y = data[n_file]
	n_edges   = len(y)
	bo 	  = np.dot(Ro.T, X)
	bi 	  = np.dot(Ri.T, X)
	B 	  = np.concatenate((bo,bi),axis=1)
	B 	  = map2angle(B)
	epoch     = len(B[:,0])
	acc_size  = round(1+epoch/EVERY_N_epoch)
	accuracy  = np.zeros(n_files*acc_size)
	accuracy[0+n_file*acc_size] = test_accuracy(B,theta_learn,y)
	for i in range(epoch):
		error 	    = TTN_edge_forward(B[i],theta_learn) - y[i]
		loss  	    = error**2
		theta_learn = TTN_edge_back(B[i],theta_learn,lr,error,y[i])
		#print('Epoch: ' + str(i) + ' Loss: ' + str(abs(loss)))
		if (i%EVERY_N_epoch==(EVERY_N_epoch-1)):
			accuracy[round((i+1)/EVERY_N_epoch) + n_file*acc_size]=test_accuracy(B,theta_learn,y)
			print('File: ' + str(n_file+1) + ' - ' + str(100*i/epoch) + '% DONE!')
# Plot the results		
for i in range(len(accuracy)):
	plt.scatter(i*EVERY_N_epoch,accuracy[i])
#plt.show()
plt.savefig('png/Accuracy.png')
print(theta_learn)
