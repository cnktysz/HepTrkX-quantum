import pennylane as qml 
from pennylane import numpy as np
import tensorflow as tf

dev1 = qml.device("default.qubit", wires=8)
@qml.qnode(dev1,interface='tf')
def TTN_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	
	# STATE PREPARATION
	for i in range(8):
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
	# Second Layer
	qml.RY(theta_learn[8],wires=1)
	qml.RY(theta_learn[9],wires=2)
	qml.CNOT(wires=[1,2])
	qml.RY(theta_learn[10],wires=5)
	qml.RY(theta_learn[11],wires=6)
	qml.CNOT(wires=[6,5])
	# Third Layer
	qml.RY(theta_learn[12],wires=2)
	qml.RY(theta_learn[13],wires=5)
	qml.CNOT(wires=[2,5])
	#Last Layer
	qml.RY(theta_learn[14],wires=5)		
	return qml.expval(qml.PauliZ(wires=5))
#################################################
dev2 = qml.device("default.qubit", wires=12)
@qml.qnode(dev2,interface='tf')
def TTN_node_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	
	# STATE PREPARATION
	for i in range(12):
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
	# Second Layer
	qml.RY(theta_learn[12],wires=1)
	qml.RY(theta_learn[13],wires=2)
	qml.CNOT(wires=[1,2])
	qml.RY(theta_learn[14],wires=5)
	qml.RY(theta_learn[15],wires=6)
	qml.CNOT(wires=[6,5])
	qml.RY(theta_learn[16],wires=9)
	qml.RY(theta_learn[17],wires=10)
	qml.CNOT(wires=[10,9])	
	# Third Layer
	qml.RY(theta_learn[18],wires=2)
	qml.RY(theta_learn[19],wires=5)
	qml.CNOT(wires=[2,5])	
	# Forth Layer
	qml.RY(theta_learn[20],wires=5)
	qml.RY(theta_learn[21],wires=9)
	qml.CNOT(wires=[5,9])
	# Last Layer
	qml.RY(theta_learn[22],wires=4)		

	return qml.expval(qml.PauliZ(wires=9))
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
		out = tf.constant(2*np.pi*(1-TTN_node_forward(node_array[i,:],theta_learn))/2.,dtype=tf.float64)
		outputs.append(out)
	return tf.stack(outputs)
#################################################
class EdgeNet(tf.keras.layers.Layer):
	def __init__(self,hid_dim=1,name='EdgeNet'):
		super(EdgeNet, self).__init__(name=name)
		# can only work with hid_dim = 1 at the moment
		self.theta_learn = tf.Variable(tf.random.uniform(shape=[15,],minval=0,maxval=np.pi*2,dtype=tf.float64))

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
		self.theta_learn = tf.Variable(tf.random.uniform(shape=[23,],minval=0,maxval=np.pi*2,dtype=tf.float64))

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
		self.num_outputs = num_outputs
		self.layer = tf.keras.layers.Dense(num_outputs,input_shape=(3,),activation='sigmoid')

	def call(self, arr):
		return self.layer(arr)*2*np.pi # to map it 0-2PI
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
			H = tf.concat([H[:,None],X],axis=1)
			e = self.EdgeNet(H, Ri, Ro)
		return e
#################################################
