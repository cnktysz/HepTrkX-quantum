import pennylane as qml 
from pennylane import numpy as np
import tensorflow as tf
from tools.tools import get_params
import pennylane_qulacs
import os
##################################################################################################
# Use default.qubit for default pennylane simulation
# use tf.interface for TF integration
USE_GPU = (os.environ["CUDA_VISIBLE_DEVICES"] != '-1')

dev1 = qml.device("qulacs.simulator", wires=8, gpu=USE_GPU)
@qml.qnode(dev1,interface='tf')
def TTN_edge_forward(edge_array,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	# STATE PREPARATION
	for i in range(8):
		qml.RY(edge_array[i],wires=i)
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
	# return expectation value of the circuit
	return qml.expval(qml.PauliZ(wires=5))
##################################################################################################
# Use default.qubit for default pennylane simulation
# use tf.interface for TF integration
dev2 = qml.device("qulacs.simulator", wires=12, gpu=USE_GPU)
@qml.qnode(dev2,interface='tf')
def TTN_node_forward(node_array,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	# STATE PREPARATION
	for i in range(12):
		qml.RY(node_array[i],wires=i)
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
	# return expectation value of the circuit
	return qml.expval(qml.PauliZ(wires=9))
##################################################################################################
def edge_forward(edge_array,theta_learn):
	# executes TTN_edge circuit for each edge in edge_array
	# To Do: can parallize the for loop
	outputs = []
	for i in range(len(edge_array[:,0])):
		out = tf.constant((1-TTN_edge_forward(edge_array[i,:],theta_learn))/2.,dtype=tf.float64)
		outputs.append(out)
	return tf.stack(outputs) # output is between [0,1]
##################################################################################################
def node_forward(node_array,theta_learn):
	# executes TTN_node circuit for each node in node_array
	# To Do: can parallize the for loop
	outputs = []
	for i in range(len(node_array[:,0])):
		out = tf.constant(2*np.pi*(1-TTN_node_forward(node_array[i,:],theta_learn))/2.,dtype=tf.float64)
		outputs.append(out)
	return tf.stack(outputs) # output is between [0,2*pi]
##################################################################################################
class EdgeNet(tf.keras.layers.Layer):
	def __init__(self, config, name='EdgeNet'):
		super(EdgeNet, self).__init__(name=name)
		# can only work with hid_dim = 1
		# read parameters of the network from file
		# params are created using tools/init_params.py
		#self.theta_learn = tf.Variable(get_params('EN',config)[0])
		self.theta_learn =  tf.Variable(tf.random.uniform(shape=[15,],minval=0,maxval=np.pi*2,dtype=tf.float64))
	def call(self,X, Ri, Ro):
		bo = tf.matmul(Ro,X,transpose_a=True)
		bi = tf.matmul(Ri,X,transpose_a=True)
		# Shape of B = N_edges x 6 (2x (3 coordinates))
		# each row consists of two node that are possibly connected.
		B  = tf.concat([bo, bi], axis=1)  
		return edge_forward(B,self.theta_learn)
##################################################################################################
class NodeNet(tf.keras.layers.Layer):
	def __init__(self, config, name='NodeNet'):
		super(NodeNet, self).__init__(name=name)
		# can only work with hid_dim = 1
		# read parameters of the network from file
		# params are created using tools/init_params.py
		#self.theta_learn = tf.Variable(get_params('NN',config)[0])
		self.theta_learn =  tf.Variable(tf.random.uniform(shape=[23,],minval=0,maxval=np.pi*2,dtype=tf.float64))
	def call(self, X, e, Ri, Ro):
		bo  = tf.matmul(Ro, X, transpose_a=True) 
		bi  = tf.matmul(Ri, X, transpose_a=True) 
		Rwo = tf.math.multiply(Ro,e)
		Rwi = tf.math.multiply(Ri,e)
		mi = tf.matmul(Rwi, bo)
		mo = tf.matmul(Rwo, bi)
		# Shape of M = N_nodes x 9 (3x (3 coordinates))
		# each row consists of a node and its 2 possible neigbours
		M = tf.concat([mi, mo, X], axis=1)
		return node_forward(M,self.theta_learn)
##################################################################################################
class InputNet(tf.keras.layers.Layer):
	def __init__(self, config, name):
		super(InputNet, self).__init__(name=name)
		self.num_outputs = config['hid_dim'] # num_outputs = number of hidden dimensions
		# read parameters of the network from file
		# params are created using tools/init_params.py
		#init = tf.constant_initializer(get_params('IN',config)[0])
		# setup a Dense layer with the given config
		self.layer = tf.keras.layers.Dense(self.num_outputs,input_shape=(3,),activation='sigmoid')
	def call(self, arr):
		return self.layer(arr)*2*np.pi # to map to output to [0,2*pi]
##################################################################################################
class GNN(tf.keras.Model):
	def __init__(self, config):
		# Network definitions here
		super(GNN, self).__init__(name='GNN')
		self.InputNet = InputNet(config = config, name='InputNet')
		self.EdgeNet  = EdgeNet(config  = config, name='EdgeNet')
		self.NodeNet  = NodeNet(config  = config, name='NodeNet')
		self.n_iters  = config['n_iters']
	
	def call(self, graph_array):
		X,Ri,Ro = graph_array                   # decompose the graph array
		H = self.InputNet(X)                    # execute InputNet to produce hidden dimensions
		H = tf.concat([H,X],axis=1)             # add new dimensions to original X matrix
		for i in range(self.n_iters):           # recurrent iteration of the network
			e = self.EdgeNet(H, Ri, Ro)         # execute EdgeNet
			H = self.NodeNet(H, e, Ri, Ro)      # execute NodeNet using the output of EdgeNet
			H = tf.concat([H[:,None],X],axis=1) # update H with the output of NodeNet
		e = self.EdgeNet(H, Ri, Ro)             # execute EdgeNet one more time to obtain edge predictions
		return e                                # return edge prediction array
##################################################################################################
