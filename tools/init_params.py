import tensorflow as tf 
import csv
import numpy as np

def log_list(params,filename):
	with open(log_dir + filename, 'w') as f:
		for item in params:
			f.write('%.15f, ' %item)


log_dir = 'params/'

input_params = tf.random.uniform(shape=[3,],dtype=tf.float32)
node_params = tf.random.uniform(shape=[23,],minval=0,maxval=np.pi*2,dtype=tf.float64)
edge_params = tf.random.uniform(shape=[15,],minval=0,maxval=np.pi*2,dtype=tf.float64)

log_list(input_params,'input_params.csv')
log_list(edge_params,'edge_params.csv')
log_list(node_params,'node_params.csv')






#init = tf.constant_initializer(YOUR_WEIGHT_MATRIX)
#l1 = tf.layers.dense(X, o, kernel_initializer=init)