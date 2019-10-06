# Author: Cenk Tüysüz
# Date: 29.08.2019
# First attempt to test QuantumEdgeNetwork
# Run this code the train and test the network

import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from datasets.hitgraphs import get_datasets
import sys, time
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
	backend = Aer.get_backend('qasm_simulator')
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
def get_loss_and_gradient(edge_array,y,theta_learn,class_weight,loss_array,gradient_array,update_array):
	local_loss     = 0.
	local_gradient = np.zeros(len(theta_learn))
	local_update   = np.zeros(len(theta_learn))
	for i in range(len(edge_array)):
		error          = (TTN_edge_forward(edge_array[i],theta_learn) - y[i])
		loss           = (error**2)*class_weight[int(y[i])]
		local_loss     += loss
		gradient       = TTN_edge_back(edge_array[i],theta_learn)*class_weight[int(y[i])]
		local_gradient += gradient
		local_update   += 2*error*gradient
	loss_array.append(local_loss)
	gradient_array.append(local_gradient)
	update_array.append(local_update)
def get_accuracy(edge_array,y,theta_learn,class_weight,error_array):
	total_acc     = 0.
	for i in range(len(edge_array)):
		total_acc += 1 - abs(TTN_edge_forward(edge_array[i],theta_learn)*class_weight[int(y[i])] - y[i])
	error_array.append(total_acc)
def train(B,theta_learn,y):
	jobs         = []
	n_threads    = 28*2
	n_edges      = len(y)
	n_feed       = n_edges//n_threads
	n_class      = [n_edges - sum(y), sum(y)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]
	# RESET variables
	manager        = multiprocessing.Manager()
	loss_array     = manager.list()
	gradient_array = manager.list()
	update_array   = manager.list()
	# Learning variables
	lr = 1
	# RUN Multithread training
	for thread in range(n_threads):
		start = thread*n_feed
		end   = (thread+1)*n_feed
		if thread==(n_threads-1):   
			p = multiprocessing.Process(target=get_loss_and_gradient,args=(B[start:,:],y[start:],theta_learn,class_weight,loss_array,gradient_array,update_array,))
		else:
			p = multiprocessing.Process(target=get_loss_and_gradient,args=(B[start:end,:],y[start:end],theta_learn,class_weight,loss_array,gradient_array,update_array,))   
		jobs.append(p)
		p.start()
	# WAIT for jobs to finish
	for proc in jobs: 
		proc.join()
		#print('Thread ended')
			
	total_loss     = sum(loss_array)
	total_gradient = sum(gradient_array)
	total_update   = sum(update_array)
	## UPDATE WEIGHTS
	average_loss     = total_loss/n_edges
	average_gradient = total_gradient/n_edges
	average_update   = total_update/n_edges
	theta_learn       = (theta_learn - lr*average_update)%(2*np.pi)
	with open('logs/log_gradients.csv', 'a') as f:
			for item in average_update:
				f.write('%.4f, ' % item)
			f.write('\n')	
	return theta_learn,average_loss
def test_validation(valid_data,theta_learn,n_valid):
	t_start = time.time()
	print('Starting testing the validation set')
	jobs         = []
	n_threads    = 28*2
	accuracy = 0.
	for n_test in range(n_valid):
		B,y          = preprocess(valid_data[n_test]) 
		n_edges      = len(y)
		n_feed       = n_edges//n_threads
		n_class      = [n_edges - sum(y), sum(y)]
		class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]
		# RESET variables
		manager        = multiprocessing.Manager()
		error_array     = manager.list()
		# RUN Multithread training
		for thread in range(n_threads):
			start = thread*n_feed
			end   = (thread+1)*n_feed
			if thread==(n_threads-1):   
				p = multiprocessing.Process(target=get_accuracy,args=(B[start:,:],y[start:],theta_learn,class_weight,error_array,))
			else:
				p = multiprocessing.Process(target=get_accuracy,args=(B[start:end,:],y[start:end],theta_learn,class_weight,error_array,))   
			jobs.append(p)
			p.start()
		# WAIT for jobs to finish
		for proc in jobs: 
			proc.join()
		accuracy += (sum(error_array)/n_edges) / n_valid
	with open('logs/log_validation.csv', 'a') as f:
				f.write('%.4f\n' % accuracy)
	duration = time.time() - t_start
	print('Validation Accuracy: %.4f, Elapsed: %dm%ds' %(accuracy*100, duration/60, duration%60))
	return accuracy
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
	theta_learn = np.random.rand(n_param)*np.pi*2 / np.sqrt(n_param)
	input_dir   = 'data/hitgraphs_big'  
	n_files     = 16*100
	n_valid     = int(n_files * 0.1)
	n_train     = n_files - n_valid	
	n_epoch     = 1
	TEST_every  = 50
	train_data, valid_data = get_datasets(input_dir, n_train, n_valid)
	loss_log = np.zeros(n_files*n_epoch)
	theta_log = np.zeros((n_files*n_epoch,11))
	valid_accuracy = np.zeros(int((n_files*0.9 // TEST_every )*n_epoch) + 2)
	valid_accuracy[0] = test_validation(valid_data,theta_learn,n_valid)
	print('Training is starting!')
	for epoch in range(n_epoch): 
		for n_file in range(n_files):
			t0 = time.time()

			B, y = preprocess(train_data[n_file])
			theta_learn,loss_log[n_file*(epoch+1)] = train(B,theta_learn,y)
			theta_log[n_file*(epoch+1),:] = theta_learn   
			t = time.time() - t0
			# Log 
			with open('logs/log_theta.csv', 'a') as f:
				for item in theta_learn:
					f.write('%.4f,' % item)
				f.write('\n')
			with open('logs/log_loss.csv', 'a') as f:
				f.write('%.4f\n' % loss_log[n_file])
			with open('logs/summary.csv', 'a') as f:
				f.write('Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds\n' % (epoch+1, n_file+1, loss_log[n_file*(epoch+1)],t / 60, t % 60) )
			print("Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" % (epoch+1, n_file+1, loss_log[n_file*(epoch+1)],t / 60, t % 60) )
			# Test validation data
			if (n_file+1)%TEST_every==0:
				valid_accuracy[(n_file+1)//TEST_every] = test_validation(valid_data,theta_learn,n_valid)
				t = time.time() - t0

	valid_accuracy[-1] = test_validation(valid_data,theta_learn)
	print('Training Complete')

	
