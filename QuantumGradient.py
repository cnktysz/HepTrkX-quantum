# Author: Cenk Tüysüz
# Date: 29.08.2019
# First attempt to test QuantumEdgeNetwork
# Run this code the train and test the network

import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from datasets.hitgraphs import HitGraphDataset
import sys
import multiprocessing


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
	result = execute(circuit, backend, shots=1000).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/1000
	return(out)

def TTN_edge_back(input_,theta_learn):
	# This function calculates the gradients for all learning 
	# variables numerically and updates them accordingly.
	# TODO: need to choose epsilon properly
	epsilon = np.pi/2 # to take derivative
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
		gradient[i] = (out_plus-out_minus)/2
		## Bring theta to its original value
		theta_learn[i] = (theta_learn[i] + epsilon)%(2*np.pi)
	
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

def test_accuracy(theta_learn):
	# This function only test the accuracy over a very limited set of data
	# due to time constraints
	# TODO: Need to test properly
	#input_dir = '/home/cenktuysuz/MyRepos/HepTrkX-quantum/data/hitgraphs'
	#input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs_big'
	input_dir = '\data\hitgraphs_big'
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

def get_loss_and_gradient(edge_array,y,theta_learn,class_weight,loss_array,gradient_array):
	# TODO: Need to add weighted loss
	local_loss = 0
	local_gradients = np.zeros(len(theta_learn))
	#print('Edge Array Size: ' + str(len(edge_array)))
	for i in range(len(edge_array)):
		error 	    = TTN_edge_forward(edge_array[i],theta_learn) - y[i]
		loss  	    = (error**2)*class_weight[int(y[i])]
		local_loss = local_loss + loss
		local_gradients = local_gradients + 2*error*TTN_edge_back(edge_array[i],theta_learn)
	loss_array.append(local_loss)
	gradient_array.append(local_gradients)
	#print(loss_array)
	#print(gradient_array)

def train(B,theta_learn,y):
	jobs 	     = []
	n_threads    = 1
	n_edges      = len(y)
	n_feed       = n_edges//n_threads
	n_class1     = sum(y)
	n_class0     = n_edges - n_class1
	w_class1     = 1 - n_class1/n_edges ## Need a better way to weight classes
	w_class0     = 1 - n_class0/n_edges
	class_weight = [w_class0,w_class1]
	# RESET variables
	manager = multiprocessing.Manager()
	loss_array = manager.list()
	gradient_array = manager.list()
	# Learning variables
	lr = 1
	# RUN Multithread training
	#print('Total edge: ' + str(n_edges))
	for thread in range(n_threads):
		start = thread*n_feed
		end   = (thread+1)*n_feed
		if thread==(n_threads-1): 	
			p = multiprocessing.Process(target=get_loss_and_gradient,args=(B[start:,:],y[start:],theta_learn,class_weight,loss_array,gradient_array,))
		else:
			p = multiprocessing.Process(target=get_loss_and_gradient,args=(B[start:end,:],y[start:end],theta_learn,class_weight,loss_array,gradient_array,))	
		jobs.append(p)
		p.start()
		#print('Thread: ' + str(thread) + ' started')

	# WAIT for jobs to finish
	for proc in jobs: 
		proc.join()
		#print('Thread ended')
			
	total_loss = sum(loss_array)
	total_gradient = sum(gradient_array)
	## UPDATE WEIGHTS
	average_loss = total_loss/n_edges
	average_gradient = total_gradient/n_edges
	print('Average Loss: ' + str(average_loss))
	print('Gradient averages' + str(average_gradient))
	theta_learn = (theta_learn - lr*average_gradient)%(2*np.pi)
	print('Update Angles :' + str(theta_learn))
	return theta_learn,average_loss

def gradient(edge_array,theta_learn,y):
	error 	    = TTN_edge_forward(edge_array,theta_learn) - y
	loss  	    = error**2
	gradient 	= 2*error*TTN_edge_back(edge_array,theta_learn)
	return [error, loss, gradient]
	#print(loss_array)
	#print(gradient_array)

############################################################################################
##### MAIN ######
#client = Client(processes=False, threads_per_worker=1, n_workers=8, memory_limit='2GB')

#client
if __name__ == '__main__':
	
	theta_learn = x = np.zeros(11) # start from zero
	input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs_big'
	n_files = 1
	testEVERY = 1
	accuracy = np.zeros(round(n_files/testEVERY) + 1)
	loss_log = np.zeros(n_files)
	theta_log = np.zeros((n_files,11))
	for n_file in range(n_files):

		data = HitGraphDataset(input_dir, n_files)
		X,Ro,Ri,y = data[n_file]
		bo 	  = np.dot(Ro.T, X)
		bi 	  = np.dot(Ri.T, X)
		B 	  = np.concatenate((bo,bi),axis=1)
		B 	  = map2angle(B)
		
		error_test    = np.zeros(100)
		loss_test     = np.zeros(100)
		gradient_test = np.zeros((100,11)) 
		x = [i for i  in range(100)]

		for i in range(100):
			error_test[i],loss_test[i],gradient_test[i,:] = gradient(B[0],theta_learn,y[0])

		# Plot the results	
		plt.clf()	
		_ = plt.hist(error_test,bins='auto')
		plt.xlabel('Error')
		plt.savefig('png/test/test_error.png')

		plt.clf()	
		_ = plt.hist(loss_test,bins='auto')
		plt.xlabel('Loss')
		plt.savefig('png/test/test_loss.png')

		for i in range(11):
			plt.clf()	
			_ = plt.hist(gradient_test[:,i],bins='auto')
			plt.title('Gradient of '+str(i)+'th angle')
			plt.savefig('png/test/test_gradient_'+str(i)+'.png')
	

