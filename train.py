# Calculates gradients of a pennylane quantum circuit
# using tensorflow
import sys, os, time, datetime, csv, yaml, argparse
sys.path.append(os.path.abspath(os.path.join('.')))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # use CPU
import tensorflow as tf
import numpy as np
from datasets.hitgraphs import get_datasets
from sklearn import metrics
from random import shuffle
from math import ceil
from tools.tools import *
from qnetworks.DNN import GNN
############################################################################################
def test(data,n_testing,testing='valid'):
	t_start = time.time()

	if testing=='valid':
		if os.path.isfile(config['log_dir']+'log_validation_preds.csv'):
			os.remove(config['log_dir']+'log_validation_preds.csv')
		print('Starting testing the validation set with ' + str(n_testing) + ' subgraphs!')
	if testing=='train':
		if os.path.isfile(config['log_dir']+'log_training_preds.csv'):
			os.remove(config['log_dir']+'log_training_preds.csv')
		print('Starting testing the training set with ' + str(n_testing) + ' subgraphs!')

	preds   = []
	labels  = []

	for n_test in range(n_testing):
		graph_array, labels_ = preprocess(data[n_test])
		labels = np.append(labels,labels_)
		preds  = np.append(preds,block(graph_array))
	
	print(len(labels))
	n_edges      = len(labels)
	n_class      = [n_edges - sum(labels), sum(labels)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]	
	#accuracy    = np.mean((1 - np.abs(preds - labels)) * [class_weight[int(labels[i])] for i in range(n_edges)]  )
	accuracy     = metrics.accuracy_score(labels.astype(int), preds > 0.5, class_weight)
	loss         = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels,preds) * np.array([class_weight[int(labels[i])] for i in range(n_edges)]))
	
	
	if testing=='valid':
		#log all preds
		with open(config['log_dir']+'log_validation_preds.csv', 'a') as f:
			for i in range(len(preds)):
				f.write('%.4f, %.4f\n' %(preds[i],labels[i]))
		#calcualte auc
		fpr,tpr,thresholds = metrics.roc_curve(labels.astype(int),preds,pos_label=1 )
		auc = metrics.auc(fpr,tpr)			
		#log
		with open(config['log_dir']+'log_validation.csv', 'a') as f:
				f.write('%.4f, %.4f, %.4f\n' %(accuracy,auc,loss))
		duration = time.time() - t_start
		print(str(datetime.datetime.now()) + ': Validation Loss: %.4f, Validation Acc: %.4f, Validation AUC: %.4f Elapsed: %dm%ds' %(loss, accuracy*100, auc, duration/60, duration%60))
	
	if testing=='train':
		#log all preds
		with open(config['log_dir']+'log_training_preds.csv', 'a') as f:
			for i in range(len(preds)):
				f.write('%.4f, %.4f\n' %(preds[i],labels[i]))
		#calcualte auc
		fpr,tpr,thresholds = metrics.roc_curve(labels.astype(int),preds,pos_label=1 )
		auc = metrics.auc(fpr,tpr)			
		#log
		with open(config['log_dir']+'log_training.csv', 'a') as f:
				f.write('%.4f, %.4f, %.4f\n' %(accuracy,auc,loss))
		duration = time.time() - t_start
		print(str(datetime.datetime.now()) + ': Training Loss: %.4f, Training Acc: %.4f, Training AUC: %.4f Elapsed: %dm%ds' %(loss, accuracy*100, auc, duration/60, duration%60))
############################################################################################
def preprocess(data):
	X,Ro,Ri,y  = data
	X 	       = tf.constant(map2angle(X),dtype=tf.float64)
	Ri         = tf.constant(Ri,dtype=tf.float64)
	Ro         = tf.constant(Ro,dtype=tf.float64)	
	edge_array = [X,Ri,Ro]
	return edge_array, tf.constant(y,dtype=tf.float64)
############################################################################################
def gradient(edge_array,label):
	n_edges      = len(labels)
	n_class      = [n_edges - sum(labels), sum(labels)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]	
	with tf.GradientTape() as tape:
		loss = tf.keras.losses.binary_crossentropy(label,block(edge_array)) * np.array([class_weight[int(labels[i])] for i in range(n_edges)])
		return loss, tape.gradient(loss,block.trainable_variables)
############################################################################################
def parse_args():
	parser = argparse.ArgumentParser(description='Load config file!')
	add_arg = parser.add_argument
	add_arg('config')
	return parser.parse_args()
def load_config(config_file):
	with open(args.config, 'r') as ymlfile:
		config = yaml.load(ymlfile)
		print('Printing configs: ')
		for key in config:
			print(key + ': ' + str(config[key]))
		print('Log dir: ' + config['log_dir'])
		print('Input dir: ' + config['input_dir'])
		delete_all_logs(config['log_dir'])
	# LOG the config
	with open(config['log_dir'] + 'config.yaml', 'w') as f:
		for key in config:
			f.write('%s : %s \n' %(key,str(config[key])))
	return config
############################################################################################
if __name__ == '__main__':
	tf.keras.backend.set_floatx('float64')
	tf.config.threading.set_inter_op_parallelism_threads(4)
	args = parse_args()
	config = load_config(args)
	train_list  = [i for i in range(config['n_train'])]

	##################### BEGIN TRAINING #####################   	
	train_data, valid_data = get_datasets(config['input_dir'], config['n_train'], config['n_valid'])
	print(str(datetime.datetime.now()) + ': Training is starting!')

	block = GNN(config['hid_dim'],config['n_iters'])
	opt = tf.keras.optimizers.Adam(learning_rate=config['lr'])

	#log_tensor_array(block.trainable_variables,config['log_dir'], 'log_learning_variables.csv') # Log Learning variables

	test(valid_data,config['n_valid'],testing='valid')
	test(train_data,config['n_train'],testing='train')

	for epoch in range(config['n_epoch']): 
		shuffle(train_list) # shuffle the order every epoch
		for n_step in range(config['n_train']):
			t0 = time.time()

			graph_array, labels = preprocess(train_data[train_list[n_step]])
			loss, grads = gradient(graph_array,labels)
			opt.apply_gradients(zip(grads, block.trainable_variables))
			t = time.time() - t0
			loss = tf.reduce_mean(loss)	
			# Log summary 
			with open(config['log_dir']+'summary.csv', 'a') as f:
				f.write('Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds\n' % (epoch+1, n_step+1, loss, t / 60, t % 60) )
			# Print summary
			print(str(datetime.datetime.now()) + ": Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" % (epoch+1, n_step+1, loss ,t / 60, t % 60) )
			"""
			# Log Learning variables
			log_tensor_array(block.trainable_variables,config['log_dir'], 'log_learning_variables.csv')
			"""
			# Log loss
			with open(config['log_dir'] + 'log_loss.csv', 'a') as f:
				f.write('%.4f\n' %loss)	
			"""	
			# Log gradients
			log_tensor_array(grads,config['log_dir'], 'log_grads.csv')
			"""
			# Test every TEST_every
			if (n_step+1)%config['TEST_every']==0:
					test(valid_data,config['n_valid'],testing='valid')
					test(train_data,config['n_train'],testing='train')
	##################### END TRAINING ##################### 


	





