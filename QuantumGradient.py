import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from datasets.hitgraphs import HitGraphDataset
import sys
import multiprocessing


def TTN_edge_forward(edge,theta_learn):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	backend = BasicAer.get_backend('qasm_simulator')
	q       = QuantumRegister(len(edge))
	c       = ClassicalRegister(1)
	circuit = QuantumCircuit(q,c)
	# STATE PREPARATION
	for i in range(len(edge)):
		circuit.ry(edge[i],q[i])
	# APPLY forward sequence

	circuit.ry(theta_learn[0],q[0])
	circuit.ry(theta_learn[1],q[1])
	circuit.cx(q[0],q[1])

	circuit.ry(theta_learn[2],q[2])
	circuit.ry(theta_learn[3],q[3])
	circuit.cx(q[2],q[3])

	circuit.ry(theta_learn[4],q[4])
	circuit.ry(theta_learn[5],q[5])
	circuit.cx(q[5],q[4]) # reverse the order

	circuit.ry(theta_learn[6],q[1])
	circuit.ry(theta_learn[7],q[3])
	circuit.cx(q[1],q[3])

	circuit.ry(theta_learn[8],q[3])
	circuit.ry(theta_learn[9],q[4])
	circuit.cx(q[3],q[4])

	circuit.ry(theta_learn[10],q[4])
	
	circuit.measure(q[4],c)

	#circuit.draw(filename='circuit.png')
	result = execute(circuit, backend, shots=100).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/100
	return(out)

def TTN_edge_back(input_,theta_learn):
	epsilon = np.pi/2 
	gradient = np.zeros(len(theta_learn))
	update = np.zeros(len(theta_learn))
	for i in range(len(theta_learn)):
		## Compute f(x+epsilon)
		theta_learn[i] = (theta_learn[i] + epsilon)%(2*np.pi)
		## Evaluate
		out_plus = TTN_edge_forward(input_,theta_learn)
		## Compute f(x-epsilon)
		theta_learn[i] = (theta_learn[i] - 2*epsilon)%(2*np.pi)
		## Evaluate
		out_minus = TTN_edge_forward(input_,theta_learn)
		# Compute the gradient numerically
		gradient[i] = (out_plus-out_minus)/2
		## Bring theta to its original value
		theta_learn[i] = (theta_learn[i] + epsilon)%(2*np.pi)
	
	return gradient
def map2angle(B):
	# Maps input features to 0-2PI
	n_section = 8
	r_min 	  = 0
	r_max     = 1.200
	phi_min   = -np.pi/n_section
	phi_max   = np.pi/n_section
	z_min 	  = -1.200
	z_max 	  = 1.200
	B[:,0] = 2*np.pi * (B[:,0]-r_min)/(r_max-r_min) 
	B[:,1] = 2*np.pi * (B[:,1]-phi_min)/(phi_max-phi_min) 
	B[:,2] = 2*np.pi * (B[:,2]-z_min)/(z_max-z_min) 
	B[:,3] = 2*np.pi * (B[:,3]-r_min)/(r_max-r_min) 
	B[:,4] = 2*np.pi * (B[:,4]-phi_min)/(phi_max-phi_min) 
	B[:,5] = 2*np.pi * (B[:,5]-z_min)/(z_max-z_min)
	return B
def get_loss_and_gradient(edge_array,y,theta_learn,class_weight,loss_array,gradient_array):
	# TODO: Need to add weighted loss
	local_loss = 0
	local_gradients = np.zeros(len(theta_learn))
	#print('Edge Array Size: ' + str(len(edge_array)))
	for i in range(len(edge_array)):
		error 	    = TTN_edge_forward(edge_array[i],theta_learn) - y[i]
		loss  	    = (error**2)*class_weight[int(y[i])]
		local_loss = local_loss + loss
		local_gradients = local_gradients + 2*error*TTN_edge_back(edge_array[i],theta_learn)
	loss_array.append(local_loss)
	gradient_array.append(local_gradients)
	#print(loss_array)
	#print(gradient_array)
def gradient(edge_array,theta_learn,y):
	error 	    = TTN_edge_forward(edge_array,theta_learn) - y
	loss  	    = error**2
	gradient 	= 2*error*TTN_edge_back(edge_array,theta_learn)
	return [error, loss, gradient]
	#print(loss_array)
	#print(gradient_array)

############################################################################################
##### MAIN ######
if __name__ == '__main__':
	
	theta_learn = x = np.zeros(11) # start from zero
	input_dir = '/Users/cenk/Repos/HEPTrkX-quantum/data/hitgraphs_big'
	n_files = 1
	testEVERY = 1
	accuracy = np.zeros(round(n_files/testEVERY) + 1)
	loss_log = np.zeros(n_files)
	theta_log = np.zeros((n_files,11))
	for n_file in range(n_files):

		data = HitGraphDataset(input_dir, n_files)
		X,Ro,Ri,y = data[n_file]
		bo 	  = np.dot(Ro.T, X)
		bi 	  = np.dot(Ri.T, X)
		B 	  = np.concatenate((bo,bi),axis=1)
		B 	  = map2angle(B)
		
		error_test    = np.zeros(100)
		loss_test     = np.zeros(100)
		gradient_test = np.zeros((100,11)) 
		x = [i for i  in range(100)]

		for i in range(100):
			error_test[i],loss_test[i],gradient_test[i,:] = gradient(B[0],theta_learn,y[0])

		# Plot the results	
		plt.clf()	
		_ = plt.hist(error_test,bins='auto')
		plt.xlabel('Error')
		plt.title('$\mu= $'+ str(round(error_test.mean(),3)) + ', std= ' + str(round(error_test.std(),3)))
		plt.savefig('png/test/test_error.png')

		plt.clf()	
		_ = plt.hist(loss_test,bins='auto')
		plt.title('$\mu= $'+ str(round(loss_test.mean(),3)) + ', std= ' + str(round(loss_test.std(),3)))
		plt.xlabel('Loss')
		plt.savefig('png/test/test_loss.png')

		for i in range(11):
			plt.clf()	
			_ = plt.hist(gradient_test[:,i],bins=10)
			plt.title('Gradient of '+str(i)+'th angle: ' + '$\mu= $'+ str(round(gradient_test[:,i].mean(),3)) + ', std= ' + str(round(gradient_test[:,i].std(),3)))
			plt.savefig('png/test/test_gradient_'+str(i)+'.png')
	

