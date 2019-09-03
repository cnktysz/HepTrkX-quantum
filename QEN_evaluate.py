import numpy as np
import matplotlib.pyplot as plt
from qiskit import(
	QuantumCircuit,
	QuantumRegister,
	ClassicalRegister,
	execute,
	BasicAer)
from qiskit.aqua.operator import Operator
from qiskit.aqua.components.initial_states import Zero
from qiskit.visualization import plot_histogram

from datasets.hitgraphs import HitGraphDataset
import sys


def TTN_edge_forward(B,theta_learn):

	backend = BasicAer.get_backend('qasm_simulator')
	q       = QuantumRegister(len(B))
	c       = ClassicalRegister(1)
	circuit = QuantumCircuit(q,c)
	# STATE PREPARATION
	for i in range(len(B)):
		circuit.ry(B[i],q[i])
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

	##circuit.draw(filename='png/QEN_circuit.png')
	result = execute(circuit, backend, shots=100).result()
	counts = result.get_counts(circuit)
	out    = 0
	for key in counts:
		if key=='1':
			out = counts[key]/100
	return(out)
	sys.exit()
def normalize(B):
	r_min_o   = min(B[:,0])  
	r_min_i   = min(B[:,3])  
	phi_min_o = min(B[:,1])  
	phi_min_i = min(B[:,4])  
	z_min_o   = min(B[:,2])  
	z_min_i   = min(B[:,5])  
	r_max_o   = min(B[:,0])  
	r_max_i   = max(B[:,3])  
	phi_max_o = max(B[:,1])  
	phi_max_i = max(B[:,4])  
	z_max_o   = max(B[:,2])  
	z_max_i   = max(B[:,5]) 
	r_min 	= min(r_min_o,r_min_i)
	r_max   = max(r_max_o,r_max_i)
	phi_min = min(phi_min_o,phi_min_i)
	phi_max = max(phi_max_o,phi_max_i)
	z_min 	= min(z_min_o,z_min_i)
	z_max 	= max(z_max_o,z_max_i)
	#print('r: '   + str(r_min)   + ' - ' + str(r_max))
	#print('phi: ' + str(phi_min) + ' - ' + str(phi_max))
	#print('z: '   + str(z_min)   + ' - ' + str(z_max))
	# Map between 0 - 2PI
	B[:,0] = 2*np.pi * (B[:,0]-r_min)/(r_max-r_min) 
	B[:,1] = 2*np.pi * (B[:,1]-phi_min)/(phi_max-phi_min) 
	B[:,2] = 2*np.pi * (B[:,2]-z_min)/(z_max-z_min) 
	B[:,3] = 2*np.pi * (B[:,3]-r_min)/(r_max-r_min) 
	B[:,4] = 2*np.pi * (B[:,4]-phi_min)/(phi_max-phi_min) 
	B[:,5] = 2*np.pi * (B[:,5]-z_min)/(z_max-z_min)
	return B 
def draw_sample(X, Ri, Ro, y, out,cmap='bwr_r', alpha_labels=True, figsize=(15, 7)):
    
    # Select the i/o node features for each segment
    feats_o = X[np.where(Ri.T)[1]]
    feats_i = X[np.where(Ro.T)[1]]
    edges = np.concatenate((feats_i,feats_o),axis=1)
    # Prepare the figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap(cmap)
    # Draw the hits (r, z)
    ax0.scatter(X[:,2], X[:,0], c='k')
    ax1.scatter(X[:,2], X[:,0], c='k')
    # Draw the segments
    for j in range(y.shape[0]):
        seg_args = dict(c='r', alpha=float(y[j]))
        ax0.plot([feats_o[j,2], feats_i[j,2]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
        seg_args = dict(c='r', alpha=float(round(out[j])))
        ax1.plot([feats_o[j,2], feats_i[j,2]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
    
    # Adjust axes
    ax0.set_xlabel('$z$')
    ax1.set_xlabel('$z$')
    ax0.set_ylabel('$r$')
    ax1.set_ylabel('$r$')
    plt.tight_layout()
    #plt.show()
    plt.savefig('png/QEN_output_RvsZ.png')

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap(cmap)
    
    # Draw the hits (r, phi)
    ax0.scatter(X[:,1], X[:,0], c='k')
    ax1.scatter(X[:,1], X[:,0], c='k')
    
    # Draw the segments
    for j in range(y.shape[0]):
        seg_args = dict(c='r', alpha=float(y[j]))
        ax0.plot([feats_o[j,1], feats_i[j,1]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
        seg_args = dict(c='r', alpha=float(round(out[j])))
        ax1.plot([feats_o[j,1], feats_i[j,1]],
                 [feats_o[j,0], feats_i[j,0]], '-', **seg_args)
    
    # Adjust axes
    ax0.set_xlabel('$\phi$')
    ax1.set_xlabel('$\phi$')
    ax0.set_ylabel('$r$')
    ax1.set_ylabel('$r$')
    plt.tight_layout()
    #plt.show()
    plt.savefig('png/QEN_output_RvsPhi.png')
############################################################################################
input_dir = 'data/hitgraphs'
theta_learn = [2.04537459, 0.09326556, 0.24176319, 0.43387259, 4.20878121, 3.3115133, 4.68544247, 3.84876339, 3.09176884, 4.15638835, 1.23]
data = HitGraphDataset(input_dir, 1)
X,Ro,Ri,y = data[0]
n_edges = len(y)
out = np.zeros(n_edges)
bo = np.dot(Ro.T, X)
bi = np.dot(Ri.T, X)
B = np.concatenate((bo,bi),axis=1)
B = normalize(B)
epoch=n_edges
for i in range(epoch):
	out[i] = TTN_edge_forward(B[i],theta_learn)
# Plot the results	
draw_sample(X, Ri, Ro, y, out)
