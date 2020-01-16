import numpy as np
import tensorflow as tf


#################################################
def edge_forward(edge_array,theta_learn):
	outputs = []
	for i in range(len(edge_array[:,0])):
		out = tf.constant((1-TTN_edge_forward(edge_array[i,:],theta_learn))/2.,dtype=tf.float64)
		outputs.append(out)
	return tf.stack(outputs)
#################################################
def node_forward(node_array,theta_learn):
	outputs = []
	for i in range(len(node_array[:,0])):
		out = tf.constant((1-TTN_node_forward(node_array[i,:],theta_learn))/2.,dtype=tf.float64)
		outputs.append(out)
	return tf.stack(outputs)
#################################################
class EdgeNet(tf.keras.layers.Layer):
	def __init__(self,name):
		super(EdgeNet, self).__init__(name=name)
		#self.theta_learn = tf.Variable(np.random.rand(15) * np.pi * 2,dtype=tf.float64)
		self.theta_learn = tf.Variable(tf.random.uniform(shape=[15,],minval=0,maxval=np.pi*2,dtype=tf.float64))
		self.kernel = tf.Variable(tf.random.uniform(shape=[8,1],minval=0,maxval=1.,dtype=tf.float64))
	def call(self,X, Ri, Ro):
		bo = tf.matmul(Ro,X,transpose_a=True)
		bi = tf.matmul(Ri,X,transpose_a=True)
		B  = tf.concat([bo, bi], axis=1)  
		return tf.matmul(B, self.kernel)
#################################################
class NodeNet(tf.keras.layers.Layer):
	def __init__(self,name):
		super(NodeNet, self).__init__(name=name)
		#self.theta_learn = tf.Variable(np.random.rand(23) * np.pi * 2,dtype=tf.float64)
		self.theta_learn = tf.Variable(tf.random.uniform(shape=[23,],minval=0,maxval=np.pi*2,dtype=tf.float64))
		self.kernel = tf.Variable(tf.random.uniform(shape=[12,1],minval=0,maxval=1.,dtype=tf.float64))
	def call(self, X, e, Ri, Ro):

		bo  = tf.matmul(Ro, X, transpose_a=True) # n_edge x 4
		bi  = tf.matmul(Ri, X, transpose_a=True) # n_edge x 4
	
		#Rwo = tf.multiply(Ro, tf.reshape(e,[e.shape[0],1])) # n_node x 1 
		#Rwi = tf.multiply(Ri, tf.reshape(e,[e.shape[0],1])) # n_node x 1
		Rwo = Ro * tf.transpose(e)
		Rwi = Ri * tf.transpose(e)
		mi = tf.matmul(Rwi, bo)
		mo = tf.matmul(Rwo,bi)
		M = tf.concat([mi, mo, X], axis=1)
		return tf.matmul(M, self.kernel)
#################################################
class InputNet(tf.keras.layers.Layer):
	def __init__(self, num_outputs,name):
		super(InputNet, self).__init__(name=name)
		self.num_outputs = num_outputs
		#self.kernel = tf.Variable(np.random.rand(3,num_outputs),dtype=tf.float64,trainable=True)
		self.kernel = tf.Variable(tf.random.uniform(shape=[3,self.num_outputs],minval=0,maxval=1.,dtype=tf.float64))

	def call(self, arr):
		return tf.matmul(arr, self.kernel)
#################################################
class GNN(tf.keras.Model):
	def __init__(self):
		super(GNN, self).__init__(name='GNN')
		self.InputNet = InputNet(1,name='InputNet')
		self.EdgeNet = EdgeNet(name='EdgeNet0')
		self.NodeNet = NodeNet(name='NodeNet')

	def call(self, edge_array, n_iters):
		X,Ri,Ro = edge_array
		H = self.InputNet(X) # not normalized, be careful !
		H = tf.concat([H,X],axis=1)
		e = self.EdgeNet(H, Ri, Ro)
		for i in range(n_iters):
			H = self.NodeNet(H, e, Ri, Ro)
			H = tf.concat([H,X],axis=1)
			e = self.EdgeNet(H, Ri, Ro)
		return e
#################################################
