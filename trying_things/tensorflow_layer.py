# Calculates gradients of a pennylane quantum circuit
# using tensorflow
import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from datasets.hitgraphs import get_datasets
from sklearn import metrics
from random import shuffle
from math import ceil
from tools.tools import *
from qnetworks.GNN import GNN
############################################################################################
def test(data,n_testing,testing='valid'):
	t_start = time.time()

	if testing=='valid':
		if os.path.isfile(log_dir+'log_validation_preds.csv'):
			os.remove(log_dir+'log_validation_preds.csv')
		print('Starting testing the validation set with ' + str(n_testing) + ' subgraphs!')
	if testing=='train':
		if os.path.isfile(log_dir+'log_training_preds.csv'):
			os.remove(log_dir+'log_training_preds.csv')
		print('Starting testing the training set with ' + str(n_testing) + ' subgraphs!')

	preds   = []
	labels  = []

	for n_test in range(n_testing):
		graph_array, labels_ = preprocess(data[n_test])
		labels.append(labels_)
		preds.append(block(graph_array))
		

	n_edges      = len(labels)
	n_class      = [n_edges - sum(labels), sum(labels)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]	
	accuracy     = mean((1 - abs(preds - labels)) * class_weight[int(labels[i])])
	loss         = tf.keras.losses.binary_crossentropy(labels,preds) 

	if testing=='valid':
		#log all preds
		with open(log_dir+'log_validation_preds.csv', 'a') as f:
			for i in range(len(preds)):
				f.write('%.4f, %.4f\n' %(preds[i],labels[i]))
		#calcualte auc
		fpr,tpr,thresholds = metrics.roc_curve(preds[:,1].astype(int),preds[:,0],pos_label=1 )
		auc = metrics.auc(fpr,tpr)			
		#log
		with open(log_dir+'log_validation.csv', 'a') as f:
				f.write('%.4f, %.4f, %.4f\n' %(accuracy,auc,loss))
		duration = time.time() - t_start
		print('Validation Loss: %.4f, Validation Acc: %.4f, Validation AUC: %.4f Elapsed: %dm%ds' %(loss, accuracy*100, auc, duration/60, duration%60))
	if testing=='train':
		#log all preds
		with open(log_dir+'log_training_preds.csv', 'a') as f:
			for i in range(len(preds)):
				f.write('%.4f, %.4f\n' %(preds[i],labels[i]))
			#calcualte auc
		fpr,tpr,thresholds = metrics.roc_curve(preds[:,1].astype(int),preds[:,0],pos_label=1 )
		auc = metrics.auc(fpr,tpr)			
		#log
		with open(log_dir+'log_training.csv', 'a') as f:
				f.write('%.4f, %.4f, %.4f\n' %(accuracy,auc,loss))
		duration = time.time() - t_start
		print('Training Loss: %.4f, Training Acc: %.4f, Training AUC: %.4f Elapsed: %dm%ds' %(loss, accuracy*100, auc, duration/60, duration%60))
############################################################################################
def preprocess(data):
	X,Ro,Ri,y  = data
	X[:,2]     = np.abs(X[:,2]) # correction for negative z
	X 	       = tf.constant(map2angle(X),dtype=tf.float64)
	Ri         = tf.constant(Ri,dtype=tf.float64)
	Ro         = tf.constant(Ro,dtype=tf.float64)	
	edge_array = [X,Ri,Ro]
	return edge_array, tf.constant(y,dtype=tf.float64)
############################################################################################
def gradient(block,edge_array,label):
	with tf.GradientTape() as tape:
		loss = tf.keras.losses.binary_crossentropy(label,block(edge_array))
		print('Loss: %.3f' %loss)
	return loss, tape.gradient(loss,block.trainable_variables)
############################################################################################
if __name__ == '__main__':
	tf.executing_eagerly()
	tf.keras.backend.set_floatx('float64')
	input_dir = 'data/hitgraphs_big'
	log_dir   = 'logs/tensorflow/ENE/lr_0_1/'
	delete_all_logs(log_dir)
	print('Log dir: ' + log_dir)
	print('Input dir: ' + input_dir)
	# Run variables
	n_files     = 16*100
	n_valid     = int(n_files * 0.01)
	n_train     = n_files - n_valid	
	train_list  = [i for i in range(n_train)]
	lr          = 0.1
	batch_size  = 5
	n_batch     = ceil(n_train/batch_size)  
	n_epoch     = 1
	n_threads   = 28
	TEST_every  = 50
	TEST_every2 = 200
	##################### BEGIN TRAINING #####################   	
	train_data, valid_data = get_datasets(input_dir, n_train, n_valid)
	print(str(datetime.datetime.now()) + ' Training is starting!')

	block = GNN()
	opt = tf.keras.optimizers.Adam(learning_rate=lr)

	# Log Learning variables
	log_tensor_array(block.trainable_variables,log_dir, 'log_learning_variables.csv')

	test(valid_data,n_valid,testing='valid')

	for epoch in range(n_epoch): 
		shuffle(train_list) # shuffle the order every epoch
		for n_step in range(n_train):
			t0 = time.time()

			graph_array, labels = preprocess(train_data[train_list[n_step]])
			loss, grads = gradient(block,graph_array,labels)
			opt.apply_gradients(zip(grads, block.trainable_variables))
			t = time.time() - t0

			# Log summary 
			with open(log_dir+'summary.csv', 'a') as f:
				f.write('Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds\n' % (epoch+1, n_step+1, loss, t / 60, t % 60) )
			# Print summary
			print(str(datetime.datetime.now()) + " Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" % (epoch+1, n_step+1, loss ,t / 60, t % 60) )
			# Log Learning variables
			log_tensor_array(block.trainable_variables,log_dir, 'log_learning_variables.csv')
			# Log loss
			with open(log_dir + 'log_loss.csv', 'a') as f:
				f.write('%.4f\n' %loss)	
			# Log gradients
			log_tensor_array(grads,log_dir, 'log_grads.csv')

			# Test every TEST_every
			if (n_step+1)%TEST_every==0:
					test(valid_data,n_valid,testing='valid')
	##################### END TRAINING ##################### 


	





