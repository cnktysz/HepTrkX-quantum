# Calculates gradients of a pennylane quantum circuit
# using tensorflow
import sys, os, time, datetime, csv, yaml, argparse
sys.path.append(os.path.abspath(os.path.join('.')))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # use CPU
# import scripts
from datasets.hitgraphs import get_datasets
from tools.tools import *
from qnetworks.GNN import GNN
from test import *
# import external
import tensorflow as tf
import numpy as np
from random import shuffle
############################################################################################
def gradient(edge_array,label):
	n_edges      = len(labels)
	n_class      = [n_edges - sum(labels), sum(labels)]
	class_weight = [n_edges/(n_class[0]*2), n_edges/(n_class[1]*2)]	
	with tf.GradientTape() as tape:
		loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label,block(edge_array)) * np.array([class_weight[int(labels[i])] for i in range(n_edges)]))
		return loss, tape.gradient(loss,block.trainable_variables)
############################################################################################
if __name__ == '__main__':
	tf.keras.backend.set_floatx('float64')
	tf.config.threading.set_inter_op_parallelism_threads(4)
	args = parse_args()
	config = load_config(args)

	train_list  = [i for i in range(config['n_train'])]
	delete_all_logs(config['log_dir'])
	##################### BEGIN TRAINING #####################   	
	train_data, valid_data = get_datasets(config['input_dir'], config['n_train'], config['n_valid'])
	print(str(datetime.datetime.now()) + ': Training is starting!')

	block = GNN(config['hid_dim'],config['n_iters'])
	opt = tf.keras.optimizers.Adam(learning_rate=config['lr'])

	#log_tensor_array(block.trainable_variables,config['log_dir'], 'log_learning_variables.csv') # Log Learning variables

	test_validation(config,block,valid_data)
	#test(train_data,config['n_train'],testing='train')

	for epoch in range(config['n_epoch']): 
		shuffle(train_list) # shuffle the order every epoch
		for n_step in range(config['n_train']):
			t0 = time.time()
			# update learning variables
			graph_array, labels = preprocess(train_data[train_list[n_step]])
			loss, grads = gradient(graph_array,labels)
			opt.apply_gradients(zip(grads, block.trainable_variables))
			
			t = time.time() - t0
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
					test_validation(config,block,valid_data)
					#test(train_data,config['n_train'],testing='train')
	##################### END TRAINING ##################### 


	





