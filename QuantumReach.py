import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
import sys,time
from qiskit.visualization import plot_histogram

def TTN_edge_forward(theta_learn,shots):
	# Takes the input and learning variables and applies the
	# network to obtain the output
	backend = BasicAer.get_backend('qasm_simulator')
	q       = QuantumRegister(6)
	c       = ClassicalRegister(6)
	circuit = QuantumCircuit(q,c)

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
	
	circuit.measure(q,c)

	result = execute(circuit, backend, shots=shots).result()
	return result.get_counts(circuit)

def add2dic(dic1,dic2):
	for keys in dic1:
		if keys not in dic2.keys():
			dic2[keys] = dic1[keys]
		else:
			dic2[keys] += dic1[keys]
	return dic2

total_dict = {}
theta_learn = np.zeros(11)
n_span = 3
for t0 in range(n_span):
	for t1 in range(n_span):
		for t2 in range(n_span):
			for t3 in range(n_span):
				for t4 in range(n_span):
					for t5 in range(n_span):
						for t6 in range(n_span):
							for t7 in range(n_span):
								for t8 in range(n_span):
									for t9 in range(n_span):
										for t10 in range(n_span):
											theta_learn[0] = t0 * 2*np.pi / n_span
											theta_learn[1] = t1 * 2*np.pi / n_span 
											theta_learn[2] = t2 * 2*np.pi / n_span 
											theta_learn[3] = t3 * 2*np.pi / n_span 
											theta_learn[4] = t4 * 2*np.pi / n_span  
											theta_learn[5] = t5 * 2*np.pi / n_span
											theta_learn[6] = t6 * 2*np.pi / n_span
											theta_learn[7] = t7 * 2*np.pi / n_span 
											theta_learn[8] = t8 * 2*np.pi / n_span 
											theta_learn[9] = t9 * 2*np.pi / n_span 
											theta_learn[10] = t10 * 2*np.pi / n_span  
											add2dic(TTN_edge_forward(theta_learn,1000),total_dict)
							print('t7 complete')
						print('t6 complete')
					print('t5 complete')							
				print('t4 complete')
			print('t3 complete')
		print('t2 complete')
	print('t1 complete')
				
plt.bar(total_dict.keys(),total_dict.values())
plt.savefig('png/QuantumReach.png')
plt.show()







