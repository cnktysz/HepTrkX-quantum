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
	qml.U3(theta_learn[0],theta_learn[1],theta_learn[2],wires=0)
	qml.U3(theta_learn[3],theta_learn[4],theta_learn[5],wires=1)
	qml.CNOT(wires=[0,1])
	qml.U3(theta_learn[6],theta_learn[7],theta_learn[8],wires=2)
	qml.U3(theta_learn[9],theta_learn[10],theta_learn[11],wires=3)
	qml.CNOT(wires=[3,2])
	qml.U3(theta_learn[12],theta_learn[13],theta_learn[14],wires=4)
	qml.U3(theta_learn[15],theta_learn[16],theta_learn[17],wires=5)
	qml.CNOT(wires=[4,5])
	qml.U3(theta_learn[18],theta_learn[19],theta_learn[20],wires=6)
	qml.U3(theta_learn[21],theta_learn[22],theta_learn[23],wires=7)
	qml.CNOT(wires=[7,6])
	# Second Layer
	qml.U3(theta_learn[24],theta_learn[25],theta_learn[26],wires=1)
	qml.U3(theta_learn[27],theta_learn[28],theta_learn[29],wires=2)
	qml.CNOT(wires=[1,2])
	qml.U3(theta_learn[30],theta_learn[31],theta_learn[32],wires=5)
	qml.U3(theta_learn[33],theta_learn[34],theta_learn[35],wires=6)
	qml.CNOT(wires=[6,5])
	# Third Layer
	qml.U3(theta_learn[36],theta_learn[37],theta_learn[38],wires=2)
	qml.U3(theta_learn[39],theta_learn[40],theta_learn[41],wires=5)
	qml.CNOT(wires=[2,5])
	#Last Layer
	qml.U3(theta_learn[42],theta_learn[43],theta_learn[44],wires=5)		
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
	qml.U3(theta_learn[0],theta_learn[1],theta_learn[2],wires=0)
	qml.U3(theta_learn[3],theta_learn[4],theta_learn[5],wires=1)
	qml.CNOT(wires=[0,1])
	qml.U3(theta_learn[6],theta_learn[7],theta_learn[8],wires=2)
	qml.U3(theta_learn[9],theta_learn[10],theta_learn[11],wires=3)
	qml.CNOT(wires=[3,2])
	qml.U3(theta_learn[12],theta_learn[13],theta_learn[14],wires=4)
	qml.U3(theta_learn[15],theta_learn[16],theta_learn[17],wires=5)
	qml.CNOT(wires=[4,5])
	qml.U3(theta_learn[18],theta_learn[19],theta_learn[20],wires=6)
	qml.U3(theta_learn[21],theta_learn[22],theta_learn[23],wires=7)
	qml.CNOT(wires=[7,6])
	qml.U3(theta_learn[24],theta_learn[25],theta_learn[26],wires=8)
	qml.U3(theta_learn[27],theta_learn[28],theta_learn[29],wires=9)
	qml.CNOT(wires=[8,9])
	qml.U3(theta_learn[30],theta_learn[31],theta_learn[32],wires=10)
	qml.U3(theta_learn[33],theta_learn[34],theta_learn[35],wires=11)
	qml.CNOT(wires=[11,10])
	# Second Layer
	qml.U3(theta_learn[36],theta_learn[37],theta_learn[38],wires=1)
	qml.U3(theta_learn[39],theta_learn[40],theta_learn[41],wires=2)
	qml.CNOT(wires=[1,2])
	qml.U3(theta_learn[42],theta_learn[43],theta_learn[44],wires=5)
	qml.U3(theta_learn[45],theta_learn[46],theta_learn[47],wires=6)
	qml.CNOT(wires=[6,5])
	qml.U3(theta_learn[48],theta_learn[49],theta_learn[50],wires=9)
	qml.U3(theta_learn[51],theta_learn[52],theta_learn[53],wires=10)
	qml.CNOT(wires=[10,9])	
	# Third Layer
	qml.U3(theta_learn[54],theta_learn[55],theta_learn[56],wires=2)
	qml.U3(theta_learn[57],theta_learn[58],theta_learn[59],wires=5)
	qml.CNOT(wires=[2,5])	
	# Forth Layer
	qml.U3(theta_learn[60],theta_learn[61],theta_learn[62],wires=5)
	qml.U3(theta_learn[63],theta_learn[64],theta_learn[65],wires=9)
	qml.CNOT(wires=[5,9])
	# Last Layer
	qml.U3(theta_learn[66],theta_learn[67],theta_learn[68],wires=4)		

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
	def __init__(self,name):
		super(EdgeNet, self).__init__(name=name)
		#self.theta_learn = tf.Variable(np.random.rand(15) * np.pi * 2,dtype=tf.float64)
		self.theta_learn = tf.Variable(tf.random.uniform(shape=[45,],minval=0,maxval=np.pi*2,dtype=tf.float64))

	def call(self,X, Ri, Ro):
		bo = tf.matmul(Ro,X,transpose_a=True)
		bi = tf.matmul(Ri,X,transpose_a=True)
		B  = tf.concat([bo, bi], axis=1)  
		return edge_forward(B,self.theta_learn)
#################################################
class NodeNet(tf.keras.layers.Layer):
	def __init__(self,name):
		super(NodeNet, self).__init__(name=name)
		#self.theta_learn = tf.Variable(np.random.rand(23) * np.pi * 2,dtype=tf.float64)
		self.theta_learn = tf.Variable(tf.random.uniform(shape=[69,],minval=0,maxval=np.pi*2,dtype=tf.float64))

	def call(self, X, e, Ri, Ro):

		bo  = tf.matmul(Ro, X, transpose_a=True) # n_edge x 4
		bi  = tf.matmul(Ri, X, transpose_a=True) # n_edge x 4
	
		#Rwo = tf.multiply(Ro, tf.reshape(e,[e.shape[0],1])) # n_node x 1 
		#Rwi = tf.multiply(Ri, tf.reshape(e,[e.shape[0],1])) # n_node x 1
		Rwo = tf.math.multiply(Ro,e)
		Rwi = tf.math.multiply(Ri,e)
		
		mi = tf.matmul(Rwi, bo)
		mo = tf.matmul(Rwo, bi)
		M = tf.concat([mi, mo, X], axis=1)
		return node_forward(M,self.theta_learn)
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
			H = tf.concat([H[:,None],X],axis=1)
			e = self.EdgeNet(H, Ri, Ro)
		return e
#################################################
