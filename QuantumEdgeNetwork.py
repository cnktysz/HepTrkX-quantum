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
from dask.distributed import Client, progress
import dask.array as da 
from multiprocessing import cpu_count
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import threading
import multiprocessing

## GLOBAL VARIABLES

def TTN_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	backend = BasicAer.get_backend('qasm_simulator')
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
	
	circuit.measure(q[4],c)

	#circuit.draw(filename='circuit.png')
	result = execute(circuit, backend, shots=100).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/100
	return(out)

def TTN_edge_back(input_,theta_learn):
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
	#update = lr*2*error*gradient
	#print('update:' + str(update))
	#theta_learn = (theta_learn - update)%(2*np.pi)
	return gradient
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
	n_section = 8
	r_min 	  = 0
	r_max     = 1.200
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

def test_accuracy(B,theta_learn):
	# This function only test the accuracy over a very limited set of data
	# due to time constraints
	# TODO: Need to test properly
	#input_dir = '/home/cenktuysuz/MyRepos/HepTrkX-quantum/data/hitgraphs'
	input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs_big'
	data = HitGraphDataset(input_dir, 1)
	X,Ro,Ri,y = data[0]
	bo   = np.dot(Ro.T, X)
	bi   = np.dot(Ri.T, X)
	B    = np.concatenate((bo,bi),axis=1)
	B    = map2angle(B)
	acc  = 0
	size = len(B[:,0])
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

def train(jobid,edge_array,loss_array,gradient_array):
	local_loss = 0
	local_gradients = np.zeros(len(theta_learn))
	for i in range(len(edge_array)):
		error 	    = TTN_edge_forward(edge_array[i],theta_learn) - y[i]
		loss  	    = error**2
		local_loss = local_loss + loss
		local_gradients = local_gradients + 2*error*TTN_edge_back(edge_array[i],theta_learn)
	loss_array.append(local_loss)
	gradient_array.append(local_gradients)
	#print(loss_array)
	#print(gradient_array)
############################################################################################
##### MAIN ######
#client = Client(processes=False, threads_per_worker=1, n_workers=8, memory_limit='2GB')

#client
if __name__ == '__main__':
	
	theta_learn = np.random.rand(11)*np.pi*2
	lr = 0.01
	EVERY_N_epoch = 500
	#input_dir = '/home/cenktuysuz/MyRepos/HepTrkX-quantum/data/hitgraphs'
	input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs_big'
	n_files = 16
	
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

		jobs = []
		# Use 16 threads
		n_threads =16
		n_feed = 50 #n_edges//n_threads
		# RESET variables
		manager = multiprocessing.Manager()
		loss_array = manager.list()
		gradient_array = manager.list()

		for i in range(n_threads):
			if i==(n_threads-1): # matrix shape correction
				p = multiprocessing.Process(target=train,args=(i,B[i*n_feed:(i+1)*n_feed,:],loss_array,gradient_array,))
			else:
				p = multiprocessing.Process(target=train,args=(i,B[i*n_feed:(i+1)*n_feed,:],loss_array,gradient_array,))
			jobs.append(p)
			p.start()

		# WAIT for jobs to finish
		for proc in jobs: proc.join()
			
		total_loss = sum(loss_array)
		total_gradient = sum(gradient_array)
		## UPDATE WEIGHTS
		average_loss = total_loss/n_edges
		average_gradient = total_gradient/n_edges
		print(average_loss)
		print(average_gradient)
		theta_learn = (theta_learn - lr*average_gradient)%(2*np.pi)
		print('Update Angles :' + str(theta_learn))
			
		sys.exit()
		#test_accuracy(B,theta_learn)

	# Plot the results		
	for i in range(len(accuracy)):
		plt.scatter(i*EVERY_N_epoch,accuracy[i])
	#plt.show()
	plt.savefig('png/Accuracy.png')
	print(theta_learn)
