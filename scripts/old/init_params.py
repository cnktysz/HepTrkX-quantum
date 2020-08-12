import tensorflow as tf 
import csv
import numpy as np

def log_list(params,filename):
	with open(log_dir + filename, 'w') as f:
		for item in params:
			f.write('%.15f, ' %item)
def log_tensor_list(tensor,filename):
# Log 2D tensorflow array
	with open(log_dir + filename, 'a') as f:
		for i in range(len(tensor)):
			for item in tensor[i].numpy():
				f.write('%.15f,' %item)
			f.write('\n')

log_dir = 'params/test/'

input_params = tf.random.uniform(shape=[3,2],dtype=tf.float32)
node_params = tf.random.uniform(shape=[31,],minval=0,maxval=np.pi*2,dtype=tf.float64)
edge_params = tf.random.uniform(shape=[19,],minval=0,maxval=np.pi*2,dtype=tf.float64)

log_tensor_list(input_params,'input_params.csv')
log_list(edge_params,'edge_params.csv')
log_list(node_params,'node_params.csv')






#init = tf.constant_initializer(YOUR_WEIGHT_MATRIX)
#l1 = tf.layers.dense(X, o, kernel_initializer=init)