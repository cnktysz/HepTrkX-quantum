import numpy as np
from qiskit import *
from qiskit.providers.aer import noise
def TTN_edge_forward(edge,theta_learn,shots=1000,noisy=False,draw=False):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	q       = QuantumRegister(8)
	c       = ClassicalRegister(1)
	circuit = QuantumCircuit(q,c)
	# STATE PREPARATION
	for i in range(len(edge)):
		circuit.ry(edge[i],q[i])
	# APPLY forward sequence
	# First Layer
	circuit.ry(theta_learn[0],q[0])
	circuit.ry(theta_learn[1],q[1])
	circuit.cx(q[0],q[1])
	circuit.ry(theta_learn[2],q[2])
	circuit.ry(theta_learn[3],q[3])
	circuit.cx(q[3],q[2])
	circuit.ry(theta_learn[4],q[4])
	circuit.ry(theta_learn[5],q[5])
	circuit.cx(q[4],q[5])
	circuit.ry(theta_learn[6],q[6])
	circuit.ry(theta_learn[7],q[7])
	circuit.cx(q[7],q[6])
	# Second Layer
	circuit.ry(theta_learn[8],q[1])
	circuit.ry(theta_learn[9],q[2])
	circuit.cx(q[1],q[2])
	circuit.ry(theta_learn[10],q[5])
	circuit.ry(theta_learn[11],q[6])
	circuit.cx(q[6],q[5])
	# Third Layer
	circuit.ry(theta_learn[12],q[2])
	circuit.ry(theta_learn[13],q[5])
	circuit.cx(q[2],q[5])
	# Last Layer
	circuit.ry(theta_learn[14],q[5])

	# Qasm Backend
	circuit.measure(q[5],c)

	if draw:	circuit.draw(filename='png/circuit/TTN8circuit.pdf')

	backend = Aer.get_backend('qasm_simulator')
	if noisy:
		noise_model = noise.device.basic_device_noise_model(properties)
		result = execute(circuit, backend, shots=shots,noise_model=noise_model).result()
	else:
		result = execute(circuit, backend, shots=shots).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/shots
	return(out)
