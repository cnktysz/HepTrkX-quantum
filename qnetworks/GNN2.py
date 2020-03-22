import pennylane as qml 
from pennylane import numpy as np
import tensorflow as tf
from tools.tools import get_params

dev1 = qml.device("default.qubit", wires=10)
@qml.qnode(dev1,interface='tf')
def TTN_edge_forward(edge,theta_learn):
	# STATE PREPARATION
	for i in range(10):
		qml.RY(edge[i],wires=i)
	# APPLY forward sequence
	# First Layer
	qml.RY(theta_learn[0],wires=0)
	qml.RY(theta_learn[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[2],wires=2)
	qml.RY(theta_learn[3],wires=3)
	qml.CNOT(wires=[3,2])
	qml.RY(theta_learn[4],wires=4)
	qml.RY(theta_learn[5],wires=5)
	qml.CNOT(wires=[5,4])
	qml.RY(theta_learn[6],wires=6)
	qml.RY(theta_learn[7],wires=7)
	qml.CNOT(wires=[6,7])
	qml.RY(theta_learn[8],wires=8)
	qml.RY(theta_learn[9],wires=9)
	qml.CNOT(wires=[8,9])
	# Second Layer
	qml.RY(theta_learn[10],wires=1)
	qml.RY(theta_learn[11],wires=2)
	qml.CNOT(wires=[1,2])
	qml.RY(theta_learn[12],wires=7)
	qml.RY(theta_learn[13],wires=9)
	qml.CNOT(wires=[9,7])
	# Third Layer
	qml.RY(theta_learn[14],wires=2)
	qml.RY(theta_learn[15],wires=4)
	qml.CNOT(wires=[2,4])
	# Forth Layer
	qml.RY(theta_learn[16],wires=4)
	qml.RY(theta_learn[17],wires=7)
	qml.CNOT(wires=[4,7])
	#Last Layer
	qml.RY(theta_learn[18],wires=7)		
	return qml.expval(qml.PauliZ(wires=7))
#################################################
dev2 = qml.device("default.qubit", wires=15)
@qml.qnode(dev2,interface='tf')
def TTN_node_forward(edge,theta_learn):
	# STATE PREPARATION
	for i in range(15):
		qml.RY(edge[i],wires=i)
	# APPLY forward sequence
	# First Layer
	qml.RY(theta_learn[0],wires=0)
	qml.RY(theta_learn[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[2],wires=2)
	qml.RY(theta_learn[3],wires=3)
	qml.CNOT(wires=[3,2])
	qml.RY(theta_learn[4],wires=4)
	qml.RY(theta_learn[5],wires=5)
	qml.CNOT(wires=[4,5])
	qml.RY(theta_learn[6],wires=6)
	qml.RY(theta_learn[7],wires=7)
	qml.CNOT(wires=[7,6])
	qml.RY(theta_learn[8],wires=8)
	qml.RY(theta_learn[9],wires=9)
	qml.CNOT(wires=[8,9])
	qml.RY(theta_learn[10],wires=10)
	qml.RY(theta_learn[11],wires=11)
	qml.CNOT(wires=[11,10])
	qml.RY(theta_learn[12],wires=12)
	qml.RY(theta_learn[13],wires=13)
	qml.CNOT(wires=[8,9])
	qml.RY(theta_learn[14],wires=14)
	# Second Layer
	qml.RY(theta_learn[15],wires=1)
	qml.RY(theta_learn[16],wires=2)
	qml.CNOT(wires=[1,2])
	qml.RY(theta_learn[14],wires=5)
	qml.RY(theta_learn[15],wires=6)
	qml.CNOT(wires=[6,5])
	qml.RY(theta_learn[16],wires=9)
	qml.RY(theta_learn[17],wires=10)
	qml.CNOT(wires=[9,10])	
	qml.RY(theta_learn[18],wires=13)
	qml.RY(theta_learn[19],wires=14)
	qml.CNOT(wires=[9,10])
	# Third Layer
	qml.RY(theta_learn[19],wires=2)
	qml.RY(theta_learn[20],wires=5)
	qml.CNOT(wires=[2,5])	
	qml.RY(theta_learn[21],wires=10)
	qml.RY(theta_learn[22],wires=13)
	qml.CNOT(wires=[13,10])	
	# Forth Layer
	qml.RY(theta_learn[23],wires=5)
	qml.RY(theta_learn[24],wires=10)
	# Fifth Layer
	qml.RY(theta_learn[25],wires=0)
	qml.RY(theta_learn[26],wires=5)
	qml.CNOT(wires=[0,5])	
	qml.RY(theta_learn[27],wires=10)
	qml.RY(theta_learn[28],wires=14)
	qml.CNOT(wires=[14,10])	
	# Last Layer
	qml.RY(theta_learn[29],wires=5)		
	qml.RY(theta_learn[30],wires=10)

	return qml.expval(qml.PauliZ(wires=5)), qml.expval(qml.PauliZ(wires=10))
#################################################
def edge_forward(edge_array,theta_learn):
	outputs = []
	for i in range(len(edge_array[:,0])):
		out = tf.constant((1-TTN_edge_forward(edge_array[i,:],theta_learn[0,:]))/2.,dtype=tf.float64)
		outputs.append(out)
	return tf.stack(outputs)
#################################################
def node_forward(node_array,theta_learn):
	outputs = []
	for i in range(len(node_array[:,0])):
		out = tf.constant(2*np.pi*(1-TTN_node_forward(node_array[i,:],theta_learn[0,:]))/2.,dtype=tf.float64)
		outputs.append(out)
	return tf.stack(outputs)
#################################################
class EdgeNet(tf.keras.layers.Layer):
	def __init__(self,hid_dim=1,name='EdgeNet'):
		super(EdgeNet, self).__init__(name=name)
		# can only work with hid_dim = 1 at the moment
		self.theta_learn = tf.Variable(get_params('edge'))
	def call(self,X, Ri, Ro):
		bo = tf.matmul(Ro,X,transpose_a=True)
		bi = tf.matmul(Ri,X,transpose_a=True)
		B  = tf.concat([bo, bi], axis=1)  
		return edge_forward(B,self.theta_learn)
#################################################
class NodeNet(tf.keras.layers.Layer):
	def __init__(self,hid_dim=1,name='NodeNet'):
		super(NodeNet, self).__init__(name=name)
		# can only work with hid_dim = 1 at the moment
		self.theta_learn = tf.Variable(get_params('node'))
	def call(self, X, e, Ri, Ro):
		bo  = tf.matmul(Ro, X, transpose_a=True) 
		bi  = tf.matmul(Ri, X, transpose_a=True) 
		Rwo = tf.math.multiply(Ro,e)
		Rwi = tf.math.multiply(Ri,e)
		mi = tf.matmul(Rwi, bo)
		mo = tf.matmul(Rwo, bi)
		M = tf.concat([mi, mo, X], axis=1)
		return node_forward(M,self.theta_learn)
#################################################
class InputNet(tf.keras.layers.Layer):
	def __init__(self, num_outputs, name):
		super(InputNet, self).__init__(name=name)
		self.params = tf.Variable(get_params('input'))
		#self.layer = tf.keras.layers.Dense(num_outputs,input_shape=(3,),activation='sigmoid',kernel_initializer=init, trainable=True)
	def call(self, arr):
		return tf.math.sigmoid(tf.matmul(arr, self.params))*2*np.pi
#################################################
class GNN(tf.keras.Model):
	def __init__(self, hid_dim=1, n_iters=2):
		super(GNN, self).__init__(name='GNN')

		self.InputNet = InputNet(num_outputs=hid_dim,name='InputNet')
		self.EdgeNet  = EdgeNet(hid_dim=hid_dim,name='EdgeNet')
		self.NodeNet  = NodeNet(hid_dim=hid_dim,name='NodeNet')
		self.n_iters = n_iters

	def call(self, edge_array):
		X,Ri,Ro = edge_array
		H = self.InputNet(X) 
		H = tf.concat([H,X],axis=1)
		e = self.EdgeNet(H, Ri, Ro)
		for i in range(self.n_iters):
			H = self.NodeNet(H, e, Ri, Ro)
			H = tf.concat([H,X], axis=1)
			e = self.EdgeNet(H, Ri, Ro)
		return e
#################################################
