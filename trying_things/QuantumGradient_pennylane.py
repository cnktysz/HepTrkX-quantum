# Author: Cenk Tüysüz
# Date: 29.08.2019
# First attempt to test QuantumEdgeNetwork
# Run this code the train and test the network

import pennylane as qml 
from pennylane import numpy as np

dev1 = qml.device("default.qubit", wires=6)

@qml.qnode(dev1)
def TTN_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	
	# STATE PREPARATION
	for i in range(len(edge)):
		qml.RY(edge[i],wires=i)
	# APPLY forward sequence
	qml.RY(theta_learn[0],wires=0)
	qml.RY(theta_learn[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(theta_learn[2],wires=2)
	qml.RY(theta_learn[3],wires=3)
	qml.CNOT(wires=[2,3])
	qml.RY(theta_learn[4],wires=4)
	qml.RY(theta_learn[5],wires=5)
	qml.CNOT(wires=[5,4])
	qml.RY(theta_learn[6],wires=1)
	qml.RY(theta_learn[7],wires=3)
	qml.CNOT(wires=[1,3])
	qml.RY(theta_learn[8],wires=3)
	qml.RY(theta_learn[9],wires=4)
	qml.CNOT(wires=[3,4])
	qml.RY(theta_learn[10],wires=4)
		
	return qml.expval(qml.PauliZ(wires=4))


def gradient(edge_array,y,theta_learn):
	dcircuit = qml.grad(TTN_edge_forward, argnum=1)
	grad = dcircuit(edge_array,theta_learn)
	return -grad/2


if __name__ == '__main__':
	
	theta_learn = np.array([0.59955824, 0.72293889, 0.76065828, 0.07674539, 0.22250796, 0.68152528, 0.72906506, 0.67450772, 0.51897852, 0.57968062, 0.3478734]) * 2 * np.pi
	edge_array = np.array([0.39437991, 0.58592331, 0.71819769, 0.07528009, 0.98749489, 0.21369549]) * 2 * np.pi 
	y = 0

	grad = gradient(edge_array,y,theta_learn)
	print('Gradients: ')
	for idx in range(len(grad)):
		print('Angle %d: %.2f' %(idx,grad[idx]))





