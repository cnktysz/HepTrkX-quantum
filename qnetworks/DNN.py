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
		self.network = tf.keras.Sequential([
            		tf.keras.layers.Dense(100,input_shape=(4*2,),activation='tanh'),
            		tf.keras.layers.Dense(1,input_shape=(100,),activation='sigmoid')
            		])
	def call(self,X, Ri, Ro):
		bo = tf.matmul(Ro,X,transpose_a=True)
		bi = tf.matmul(Ri,X,transpose_a=True)
		B  = tf.concat([bo, bi], axis=1)  
		return self.network(B)
#################################################
class NodeNet(tf.keras.layers.Layer):
	def __init__(self,name):
		super(NodeNet, self).__init__(name=name)
		self.network = tf.keras.Sequential([
            		tf.keras.layers.Dense(100,input_shape=(4*3,),activation='tanh'),
            		tf.keras.layers.Dense(1,activation='sigmoid')
            		])
	def call(self, X, e, Ri, Ro):

		bo  = tf.matmul(Ro, X, transpose_a=True) # n_edge x 4
		bi  = tf.matmul(Ri, X, transpose_a=True) # n_edge x 4	
		Rwo = Ro * tf.transpose(e)
		Rwi = Ri * tf.transpose(e)
		mi = tf.matmul(Rwi, bo)
		mo = tf.matmul(Rwo,bi)
		M = tf.concat([mi, mo, X], axis=1)
		return self.network(M)
#################################################
class InputNet(tf.keras.layers.Layer):
	def __init__(self, num_outputs,name):
		super(InputNet, self).__init__(name=name)
		self.num_outputs = num_outputs
		self.layer = tf.keras.layers.Dense(num_outputs,input_shape=(3,),activation='sigmoid')
	
	def call(self, arr):
		return self.layer(arr)
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
