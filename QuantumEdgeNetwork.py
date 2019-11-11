#!/usr/bin/python
# Author: Cenk Tuysuz
# Date: 29.08.2019
# First attempt to test QuantumEdgeNetwork
# Run this code the train and test the network

import numpy as np
import matplotlib.pyplot as plt
from datasets.hitgraphs import get_datasets
import sys, os, time, datetime, csv
import multiprocessing
from qnetworks.TTN import TTN_edge_forward, TTN_edge_back
from qnetworks.MERA import MERA_edge_forward, MERA_edge_back
from sklearn import metrics
from qiskit import *
########################################################
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
	B[:,5] =  (B[:,5]-z_min)/(z_max-z_min)* 2 * np.pi 
	return B
########################################################	
def binary_cross_entropy(output,label):
	return -(label*np.log(output+1e-6) + (1-label)*np.log(1-output+1e-6))
########################################################
def get_loss_and_gradient(edge_array,y,theta_learn,class_weight,loss_array,gradient_array,update_array):
	local_loss     = 0.
	local_gradient = np.zeros(len(theta_learn))
	local_update   = np.zeros(len(theta_learn))
	for i in range(len(edge_array)):
		output         = MERA_edge_forward(edge_array[i],theta_learn,properties,shots=shots)
		error          = output - y[i]
		local_loss     += binary_cross_entropy(output, y[i])*class_weight[int(y[i])]
		gradient       = MERA_edge_back(edge_array[i],theta_learn,properties,shots=shots)*class_weight[int(y[i])]
		local_gradient += gradient
		local_update   += 2*error*gradient
	loss_array.append(local_loss)
	gradient_array.append(local_gradient)
	update_array.append(local_update)
########################################################
def get_accuracy(edge_array,labels,theta_learn,class_weight,acc_array,loss_array):
	total_acc  = 0.
	total_loss = 0.
	for i in range(len(edge_array)):
		pred = MERA_edge_forward(edge_array[i],theta_learn,properties,shots=shots)
		total_acc  += (1 - abs(pred - labels[i]))*class_weight[int(labels[i])]
		total_loss += binary_cross_entropy(pred,labels[i])
		with open(log_dir+'log_validation_preds.csv', 'a') as f:
				f.write('%.4f, %.4f\n' %(pred,labels[i]))
	acc_array.append(total_acc)
	loss_array.append(total_loss)
########################################################
def train(B,theta_learn,y):
	jobs         = []
	n_edges      = len(y)
	n_feed       = n_edges//n_threads
	n_class      = [n_edges - sum(y), sum(y)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]
	# RESET variables
	manager        = multiprocessing.Manager()
	loss_array     = manager.list()
	gradient_array = manager.list()
	update_array   = manager.list()
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
	theta_learn      = (theta_learn - lr*average_update)%(2*np.pi)
	with open(log_dir+'log_gradients.csv', 'a') as f:
			for item in average_update:
				f.write('%.4f, ' % item)
			f.write('\n')	
	return theta_learn,average_loss
########################################################
def test_validation(valid_data,theta_learn,n_valid):
	if os.path.isfile(log_dir+'log_validation_preds.csv'):
		os.remove(log_dir+'log_validation_preds.csv')
	t_start = time.time()
	print('Starting testing the validation set with ' + str(n_valid) + ' subgraphs!')
	jobs     = []
	accuracy = 0.
	loss 	 = 0.
	for n_test in range(n_valid):
		B,y          = preprocess(valid_data[n_test]) 
		n_edges      = len(y)
		n_feed       = n_edges//n_threads
		n_class      = [n_edges - sum(y), sum(y)]
		class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]
		# RESET variables
		manager    = multiprocessing.Manager()
		acc_array  = manager.list()
		loss_array = manager.list()
		# RUN Multithread training
		for thread in range(n_threads):
			start = thread*n_feed
			end   = (thread+1)*n_feed
			if thread==(n_threads-1):   
				p = multiprocessing.Process(target=get_accuracy,args=(B[start:,:],y[start:],theta_learn,class_weight,acc_array,loss_array,))
			else:
				p = multiprocessing.Process(target=get_accuracy,args=(B[start:end,:],y[start:end],theta_learn,class_weight,acc_array,loss_array,))   
			jobs.append(p)
			p.start()
		# WAIT for jobs to finish
		for proc in jobs: 
			proc.join()
		accuracy += sum(acc_array)/(n_edges * n_valid)
		loss 	 += sum(loss_array)/(n_edges * n_valid)

	#read all preds
	with open(log_dir + 'log_validation_preds.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		preds = np.array(list(reader)).astype(float)
	#calcualte auc
	fpr,tpr,thresholds = metrics.roc_curve(preds[:,1].astype(int),preds[:,0],pos_label=1 )
	auc = metrics.auc(fpr,tpr)			
	#log
	with open(log_dir+'log_validation.csv', 'a') as f:
			f.write('%.4f, %.4f, %.4f\n' %(accuracy,auc,loss))
	duration = time.time() - t_start
	print('Validation Loss: %.4f, Validation Acc: %.4f, Validation AUC: %.4f Elapsed: %dm%ds' %(loss, accuracy*100, auc, duration/60, duration%60))
########################################################
def preprocess(data):
	X,Ro,Ri,y = data
	X[:,2]    = np.abs(X[:,2]) # correction for negative z
	bo        = np.dot(Ro.T, X)
	bi        = np.dot(Ri.T, X)
	B         = np.concatenate((bo,bi),axis=1)
	return map2angle(B), y
############################################################################################
##### MAIN ######
if __name__ == '__main__':
	n_param = 19
	theta_learn = np.random.rand(n_param)*np.pi*2 #/ np.sqrt(n_param)
	input_dir   = 'data/hitgraphs_big'
	log_dir     = 'logs/MERA/lr_0_1/'  
	print('Log dir: ' + log_dir)
	print('Input dir: ' + input_dir)
	provider = IBMQ.load_account()
	backends = provider.backends()
	device = provider.get_backend('ibmq_16_melbourne')
	properties = device.properties()
	shots=1000
	n_files     = 16*100
	n_valid     = int(n_files * 0.1)
	n_train     = n_files - n_valid	
	lr          = 0.1
	n_epoch     = 5
	batch_size  = 5
	n_threads   = 28
	TEST_every  = 50
	train_data, valid_data = get_datasets(input_dir, n_train, n_valid)
	test_validation(valid_data,theta_learn,n_valid)
	print(str(datetime.datetime.now()) + ' Training is starting!')
	for epoch in range(n_epoch): 
		for n_file in range(n_train):
			t0 = time.time()
			B, y = preprocess(train_data[n_file])
			theta_learn,loss = train(B,theta_learn,y)
			t = time.time() - t0
			# Log 
			with open(log_dir+'log_theta.csv', 'a') as f:
				for item in theta_learn:
					f.write('%.4f,' % item)
				f.write('\n')
			with open(log_dir+'log_loss.csv', 'a') as f:
				f.write('%.4f\n' % loss)
			with open(log_dir+'summary.csv', 'a') as f:
				f.write('Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds\n' % (epoch+1, n_file+1, loss, t / 60, t % 60) )
			print(str(datetime.datetime.now()) + " Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" % (epoch+1, n_file+1, loss ,t / 60, t % 60) )
			# Test validation data
			if (n_file+1)%TEST_every==0:
				test_validation(valid_data,theta_learn,n_valid)
		print('Epoch Complete!')
	test_validation(valid_data,theta_learn,n_valid)
	print('Training Complete!')

	
