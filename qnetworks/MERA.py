import numpy as np
from qiskit import *
from qiskit.providers.aer import noise

def MERA_edge_forward(edge,theta_learn,properties,shots=1000,noisy=False,draw=False):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	# Has 19 parameters (theta_learn)
	q       = QuantumRegister(len(edge))
	c       = ClassicalRegister(1)
	circuit = QuantumCircuit(q,c)
	# STATE PREPARATION
	#######  Seq. 0  #######
	for i in range(len(edge)):
		circuit.ry(edge[i],q[i])
	circuit.barrier()
	# APPLY forward sequence
	#######  Seq. 1  #######
	circuit.ry(theta_learn[0],q[1])
	circuit.ry(theta_learn[1],q[2])
	circuit.cx(q[1],q[2])
	circuit.ry(theta_learn[2],q[3])
	circuit.ry(theta_learn[3],q[4])
	circuit.cx(q[3],q[4])
	circuit.barrier()
	#######  Seq. 2  #######
	circuit.ry(theta_learn[4],q[0])
	circuit.ry(theta_learn[5],q[1])
	circuit.cx(q[0],q[1])
	circuit.ry(theta_learn[6],q[2])
	circuit.ry(theta_learn[7],q[3])
	circuit.cx(q[2],q[3])
	circuit.ry(theta_learn[8],q[4])
	circuit.ry(theta_learn[9],q[5])
	circuit.cx(q[5],q[4]) # reverse the order
	circuit.barrier()
	#######  Seq. 3  #######
	circuit.ry(theta_learn[10],q[1])
	circuit.ry(theta_learn[11],q[4])
	circuit.cx(q[1],q[4])
	circuit.barrier()
	#######  Seq. 4  #######
	circuit.ry(theta_learn[12],q[1])
	circuit.ry(theta_learn[13],q[2])
	circuit.cx(q[1],q[2])
	circuit.ry(theta_learn[14],q[3])
	circuit.ry(theta_learn[15],q[4])
	circuit.cx(q[4],q[3])
	circuit.barrier()
	#######  Seq. 5  #######
	circuit.ry(theta_learn[16],q[2])
	circuit.ry(theta_learn[17],q[3])
	circuit.cx(q[2],q[3])
	circuit.barrier()
	#######  Seq. 6  #######
	circuit.ry(theta_learn[18],q[3])
	# Qasm Backend
	circuit.measure(q[3],c)

	if draw:	circuit.draw(filename='png/circuit/MERAcircuit.pdf')
	
	# Qasm Backend
	circuit.measure(q[4],c)
	backend = Aer.get_backend('qasm_simulator')
	if noisy:
		noise_model = noise.device.basic_device_noise_model(properties)
		result = execute(circuit, backend, shots=shots,noise_model=noise_model).result()
	else:
		result = execute(circuit, backend, shots=shots).result()

	result = execute(circuit, backend, shots=shots).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/shots
	return(out)
def MERA_edge_back(input_,theta_learn,properties,shots=1000,noisy=False):
	# This function calculates the gradients for all learning 
	# variables numerically and updates them accordingly.
	epsilon = np.pi/2 # to take derivative
	gradient = np.zeros(len(theta_learn))
	update = np.zeros(len(theta_learn))
	for i in range(len(theta_learn)):
		## Compute f(x+epsilon)
		theta_learn[i] = (theta_learn[i] + epsilon)%(2*np.pi)
		## Evaluate
		out_plus = MERA_edge_forward(input_,theta_learn,properties,shots=shots,noisy=noisy)
		## Compute f(x-epsilon)
		theta_learn[i] = (theta_learn[i] - 2*epsilon)%(2*np.pi)
		## Evaluate
		out_minus = MERA_edge_forward(input_,theta_learn,properties,shots=shots,noisy=noisy)
		# Compute the gradient numerically
		gradient[i] = (out_plus-out_minus)/2
		## Bring theta to its original value
		theta_learn[i] = (theta_learn[i] + epsilon)%(2*np.pi)
	return gradient