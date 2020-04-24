import numpy as np
import tensorflow as tf
#################################################
class EdgeNet(tf.keras.layers.Layer):
	def __init__(self,config,name):
		super(EdgeNet, self).__init__(name=name)
		hid_dim = config['hid_dim']
		self.network = tf.keras.Sequential([
            		tf.keras.layers.Dense(hid_dim,input_shape=((hid_dim+3)*2,),activation='tanh'),
            		tf.keras.layers.Dense(1,activation='sigmoid')
            		])
	def call(self,X, Ri, Ro):
		bo = tf.matmul(Ro,X,transpose_a=True)
		bi = tf.matmul(Ri,X,transpose_a=True)
		B  = tf.concat([bo, bi], axis=1)  
		return self.network(B)
#################################################
class NodeNet(tf.keras.layers.Layer):
	def __init__(self,config,name):
		super(NodeNet, self).__init__(name=name)
		hid_dim = config['hid_dim']
		self.network = tf.keras.Sequential([
            		tf.keras.layers.Dense(hid_dim,input_shape=((hid_dim+3)*3,),activation='tanh'),
            		tf.keras.layers.Dense(hid_dim,activation='sigmoid')
            		])
	def call(self, X, e, Ri, Ro):
		bo  = tf.matmul(Ro, X, transpose_a=True) 
		bi  = tf.matmul(Ri, X, transpose_a=True) 
		Rwo = Ro * tf.transpose(e)
		Rwi = Ri * tf.transpose(e)
		mi = tf.matmul(Rwi, bo)
		mo = tf.matmul(Rwo,bi)
		M = tf.concat([mi, mo, X], axis=1)
		return self.network(M)
#################################################
class InputNet(tf.keras.layers.Layer):
	def __init__(self, config,name):
		super(InputNet, self).__init__(name=name)
		self.layer = tf.keras.layers.Dense(config['hid_dim'],input_shape=(3,),activation='tanh')
	
	def call(self, arr):
		return self.layer(arr)
#################################################
class GNN(tf.keras.Model):
	def __init__(self, config):
		super(GNN, self).__init__(name='GNN')
		self.InputNet = InputNet(config=config, name='InputNet')
		self.EdgeNet = EdgeNet(config=config, name='EdgeNet')
		self.NodeNet = NodeNet(config=config, name='NodeNet')
		self.n_iters = config['n_iters']

	def call(self, graph_array):
		X,Ri,Ro = graph_array
		H = self.InputNet(X) 
		H = tf.concat([H,X],axis=1)
		for i in range(self.n_iters):
			e = self.EdgeNet(H, Ri, Ro)
			H = self.NodeNet(H, e, Ri, Ro)
			H = tf.concat([H,X],axis=1)
		e = self.EdgeNet(H, Ri, Ro)
		return e
#################################################
