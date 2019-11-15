import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import matplotlib.pyplot as plt
import pennylane as qml 
from pennylane import numpy as np
from datasets.hitgraphs import get_datasets
import multiprocessing
from sklearn import metrics
from random import shuffle
from math import ceil

dev1 = qml.device("default.qubit", wires=6)

@qml.qnode(dev1)
def TTN_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	# STATE PREPARATION
	for i in range(len(edge)):
		qml.RY(edge[i],wires=i)
	# APPLY forward sequence
	qml.RY(theta_learn[0],wires=0)
	qml.RY(theta_learn[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[2],wires=2)
	qml.RY(theta_learn[3],wires=3)
	qml.CNOT(wires=[2,3])
	qml.RY(theta_learn[4],wires=4)
	qml.RY(theta_learn[5],wires=5)
	qml.CNOT(wires=[5,4])
	qml.RY(theta_learn[6],wires=1)
	qml.RY(theta_learn[7],wires=3)
	qml.CNOT(wires=[1,3])
	qml.RY(theta_learn[8],wires=3)
	qml.RY(theta_learn[9],wires=4)
	qml.CNOT(wires=[3,4])
	qml.RY(theta_learn[10],wires=4)
	return(qml.expval(qml.PauliZ(wires=4)))
"""
dev2 = qml.device("default.qubit", wires=6)

@qml.qnode(dev2)
def MERA_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	# STATE PREPARATION
	for i in range(len(edge)):
		qml.RY(edge[i],wires=i)
	# APPLY forward sequence
	#######  Seq. 1  #######
	qml.RY(theta_learn[0],wires=0)
	qml.RY(theta_learn[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[2],wires=3)
	qml.RY(theta_learn[3],wires=4)
	qml.CNOT(wires=[3,4])
	#######  Seq. 2  #######
	qml.RY(theta_learn[4],wires=0)
	qml.RY(theta_learn[5],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[6],wires=2)
	qml.RY(theta_learn[7],wires=3)
	qml.CNOT(wires=[2,3])
	qml.RY(theta_learn[8],wires=4)
	qml.RY(theta_learn[9],wires=5)
	qml.CNOT(wires=[5,4])
	#######  Seq. 3  #######
	qml.RY(theta_learn[10],wires=1)
	qml.RY(theta_learn[11],wires=4)
	qml.CNOT(wires=[1,4])
	#######  Seq. 4  #######
	qml.RY(theta_learn[12],wires=1)
	qml.RY(theta_learn[13],wires=2)
	qml.CNOT(wires=[1,2])
	qml.RY(theta_learn[14],wires=3)
	qml.RY(theta_learn[15],wires=4)
	qml.CNOT(wires=[4,3])
	#######  Seq. 5  #######
	qml.RY(theta_learn[16],wires=2)
	qml.RY(theta_learn[17],wires=3)
	qml.CNOT(wires=[2,3])
	#######  Seq. 5  #######
	qml.RY(theta_learn[18],wires=3)
	return(qml.expval(qml.PauliZ(wires=3)))
"""
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
def loss_fn(edge_array,label,theta_learn,class_weight,loss_array):
	loss = 0.
	for i in range(len(label)):
		output = (1 - TTN_edge_forward(edge_array[i],theta_learn))/2
		loss  += binary_cross_entropy(output,label[i]) * class_weight[int(label[i])]
	loss_array.append(loss)
def cost_fn(edge_array,y,theta_learn):
	jobs         = []
	n_edges      = len(y)
	n_feed       = n_edges//n_threads
	n_class      = [n_edges - sum(y), sum(y)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]
	manager        = multiprocessing.Manager()
	loss_array     = manager.list()
	for thread in range(n_threads):
		start = thread*n_feed
		end   = (thread+1)*n_feed
		if thread==(n_threads-1):   
			p = multiprocessing.Process(target=loss_fn,args=(edge_array[start:,:],y[start:],theta_learn,class_weight,loss_array,))
		else:
			p = multiprocessing.Process(target=loss_fn,args=(edge_array[start:end,:],y[start:end],theta_learn,class_weight,loss_array,))   
		jobs.append(p)
		p.start()
	# WAIT for jobs to finish
	for proc in jobs: 
		proc.join()
	return sum(loss_array)/n_edges
def gradient(edge_array,y,theta_learn,gradient_array):
	grad = np.zeros(len(theta_learn))
	for i in range(len(edge_array)):
		dcircuit = qml.grad(TTN_edge_forward, argnum=1)
		grad += dcircuit(edge_array[i],theta_learn)
	gradient_array.append(-grad/2)	
def grad_fn(edge_array,y,theta_learn):
	jobs         = []
	n_edges      = len(y)
	n_feed       = n_edges//n_threads
	n_class      = [n_edges - sum(y), sum(y)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]
	manager        = multiprocessing.Manager()
	gradient_array  = manager.list()
	for thread in range(n_threads):
		start = thread*n_feed
		end   = (thread+1)*n_feed
		if thread==(n_threads-1):   
			p = multiprocessing.Process(target=gradient,args=(edge_array[start:,:],y[start:],theta_learn,gradient_array,))
		else:
			p = multiprocessing.Process(target=gradient,args=(edge_array[start:end,:],y[start:end],theta_learn,gradient_array,))   
		jobs.append(p)
		p.start()
	# WAIT for jobs to finish
	for proc in jobs: 
		proc.join()

	avg_grad = sum(gradient_array)/n_edges
	with open(log_dir+'log_gradient.csv', 'a') as f:
		for item in avg_grad:
			f.write('%.4f,' % item)
		f.write('\n')

	return avg_grad
def binary_cross_entropy(output,label):
	return -( label*np.log(output+1e-6) + (1-label)*np.log(1-output+1e-6) )
def test(data,theta_learn,n_testing,testing='valid'):
	t_start = time.time()

	if testing=='valid':
		if os.path.isfile(log_dir+'log_validation_preds.csv'):
			os.remove(log_dir+'log_validation_preds.csv')
		print('Starting testing the validation set with ' + str(n_testing) + ' subgraphs!')
	if testing=='train':
		if os.path.isfile(log_dir+'log_training_preds.csv'):
			os.remove(log_dir+'log_training_preds.csv')
		print('Starting testing the training set with ' + str(n_testing) + ' subgraphs!')

	jobs     = []
	accuracy = 0.
	loss 	 = 0.
	for n_test in range(n_testing):
		B,y          = preprocess(data[n_test]) 
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
				p = multiprocessing.Process(target=get_accuracy,args=(B[start:,:],y[start:],theta_learn,class_weight,acc_array,loss_array,testing,))
			else:
				p = multiprocessing.Process(target=get_accuracy,args=(B[start:end,:],y[start:end],theta_learn,class_weight,acc_array,loss_array,testing,))   
			jobs.append(p)
			p.start()
		# WAIT for jobs to finish
		for proc in jobs: 
			proc.join()
		accuracy += sum(acc_array)/(n_edges * n_testing)
		loss 	 += sum(loss_array)/(n_edges * n_testing)

	if testing=='valid':
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
	if testing=='train':
		#read all preds
		with open(log_dir + 'log_training_preds.csv', 'r') as f:
			reader = csv.reader(f, delimiter=',')
			preds = np.array(list(reader)).astype(float)
			#calcualte auc
		fpr,tpr,thresholds = metrics.roc_curve(preds[:,1].astype(int),preds[:,0],pos_label=1 )
		auc = metrics.auc(fpr,tpr)			
		#log
		with open(log_dir+'log_training.csv', 'a') as f:
				f.write('%.4f, %.4f, %.4f\n' %(accuracy,auc,loss))
		duration = time.time() - t_start
		print('Training Loss: %.4f, Training Acc: %.4f, Training AUC: %.4f Elapsed: %dm%ds' %(loss, accuracy*100, auc, duration/60, duration%60))

def get_accuracy(edge_array,labels,theta_learn,class_weight,acc_array,loss_array,mode):
	total_acc  = 0.
	total_loss = 0.
	for i in range(len(edge_array)):
		pred = (1 - TTN_edge_forward(edge_array[i],theta_learn))/2
		total_acc  += (1 - abs(pred - labels[i]))*class_weight[int(labels[i])]
		total_loss += binary_cross_entropy(pred,labels[i])*class_weight[int(labels[i])]
		if mode=='valid':
			with open(log_dir+'log_validation_preds.csv', 'a') as f:
				f.write('%.4f, %.4f\n' %(pred,labels[i]))
		if mode=='train':
			with open(log_dir+'log_training_preds.csv', 'a') as f:
				f.write('%.4f, %.4f\n' %(pred,labels[i]))
	acc_array.append(total_acc)
	loss_array.append(total_loss)
def preprocess(data):
	X,Ro,Ri,y = data
	X[:,2]    = np.abs(X[:,2]) # correction for negative z
	bo        = np.dot(Ro.T, X)
	bi        = np.dot(Ri.T, X)
	B         = np.concatenate((bo,bi),axis=1)
	return map2angle(B), y
def delete_all_logs(log_dir):
	log_list = os.listdir(log_dir)
	for item in log_list:
		if item.endswith('.csv'):
			os.remove(log_dir+item)
			print(str(datetime.datetime.now()) + ' Deleted old log: ' + log_dir+item)
if __name__ == '__main__':
	n_param = 19
	theta_learn = np.random.rand(n_param) * np.pi * 2 
	input_dir = 'data/hitgraphs_big'
	log_dir   = 'logs/pennylane/MERA/lr_0_1/'
	delete_all_logs(log_dir)
	print('Log dir: ' + log_dir)
	print('Input dir: ' + input_dir)
	# Run variables
	n_files     = 16*100
	n_valid     = int(n_files * 0.1)
	n_train     = n_files - n_valid	
	train_list  = [i for i in range(n_train)]
	lr          = 0.1
	batch_size  = 5
	n_batch     = ceil(n_train/batch_size)  
	n_epoch     = 5
	n_threads   = 28
	TEST_every  = 50
	TEST_every2  = 200
	#####################   BEGIN   #####################   	
	train_data, valid_data = get_datasets(input_dir, n_train, n_valid)
	test(valid_data,theta_learn,n_valid,'valid')
	#test(train_data,theta_learn,n_train,'train')
	print(str(datetime.datetime.now()) + ' Training is starting!')
	#opt = qml.AdamOptimizer(stepsize=lr, beta1=0.9, beta2=0.99,eps=1e-08)
	opt = qml.GradientDescentOptimizer(stepsize=lr)
	for epoch in range(n_epoch): 
		shuffle(train_list) # shuffle the order every epoch
		for n_step in range(n_train):
			t0 = time.time()
			B, y = preprocess(train_data[train_list[n_step]])
			# Update learning variables
			theta_learn = opt.step(lambda v: cost_fn(B,y,v),theta_learn,lambda z: grad_fn(B,y,theta_learn))
			theta_learn = theta_learn % (2*np.pi)
			
			loss = cost_fn(B,y,theta_learn) # Need this to log loss
			t = time.time() - t0

			# Log the result every update  
			with open(log_dir+'summary.csv', 'a') as f:
				f.write('Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds\n' % (epoch+1, n_step+1, loss, t / 60, t % 60) )
			
			print(str(datetime.datetime.now()) + " Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" % (epoch+1, n_step+1, loss ,t / 60, t % 60) )
			
			with open(log_dir + 'log_theta.csv', 'a') as f:
				for item in theta_learn:
					f.write('%.4f,' % item)
				f.write('\n')
			
			with open(log_dir + 'log_loss.csv', 'a') as f:
				f.write('%.4f\n' % loss)	
			
			# Test every TEST_every
			if (n_step+1)%TEST_every==0:
					test(valid_data,theta_learn,n_valid,'valid')
		
		print('Epoch Complete!')

	test(valid_data,theta_learn,n_valid,'valid')
	#test(train_data,theta_learn,n_train,'train')
	print('Training Complete!')


