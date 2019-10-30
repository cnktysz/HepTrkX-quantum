"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch.autograd import Function
import torch.optim as optim
from tqdm import tqdm
from qiskit import Aer
from QuantumEdgeNetwork import TTN_edge_forward
import numpy as np
from qiskit import QuantumRegister,QuantumCircuit,ClassicalRegister,execute
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from datasets.hitgraphs import get_datasets
import sys 


def to_numbers(tensor_list):
    num_list = []
    for tensor in tensor_list: 
        num_list += [tensor.item()]
    return num_list

class QiskitCircuit():
    def __init__(self,edge,theta_learn,shots):
        self.shots   = shots        

    def run(self,edge,theta_learn,shots):
        out = []
        for n_edge in range(len(edge)):
            out.append(TTN(to_numbers(edge[n_edge]),theta_learn,shots))
        return torch.tensor(out)


class TorchCircuit(Function):
    @staticmethod
    def forward(ctx,edge,theta_learn):
        #if not hassattr(ctx, 'QiskitCircuit'):
        ctx.QiskitCirc = QiskitCircuit(edge,theta_learn,shots=1000)
        out = ctx.QiskitCirc.run(edge,to_numbers(theta_learn),shots=1000)
        ctx.save_for_backward(out,edge,theta_learn)
        return out
    @staticmethod
    def backward(ctx,grad_output):
        eps=np.pi/2
        forward_tensor, edge, theta_learn = ctx.saved_tensors
        input_numbers = to_numbers(theta_learn)
        gradient = torch.zeros(11)
        for k in range(len(theta_learn)):
            input_eps = input_numbers
            input_eps[k] = input_numbers[k] + eps
            out1 = ctx.QiskitCirc.run(edge,input_eps,shots=1000)
            input_eps[k] = input_numbers[k] - eps
            out2 = ctx.QiskitCirc.run(edge,input_eps,shots=1000)
            gradient_result = (out1 - out2)/2
            gradient[k] = torch.mean(gradient_result)# take mean
        return None, gradient
def TTN(edge,theta_learn,shots):
    n_qubits = 6
    q        = QuantumRegister(n_qubits,'q')
    c        = ClassicalRegister(1,'c')
    circuit  = QuantumCircuit(q,c)
    # STATE PREPARATION
    for i in range(n_qubits):
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

    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend, shots=shots).result()
    counts = result.get_counts(circuit)
    for key in counts:
        if counts['1']:
            return counts['1']/shots
        else:
            return 0.

def map2angle(B):
    # Maps input features to 0-2PI
    r_min     = 0.
    r_max     = 1.
    phi_min   = -1.
    phi_max   = 1.
    z_min     = 0.
    z_max     = 1.2
    B[:,0] =  (B[:,0]-r_min)/(r_max-r_min)       * 2 * np.pi 
    B[:,1] =  (B[:,1]-phi_min)/(phi_max-phi_min) * 2 * np.pi 
    B[:,2] =  (B[:,2]-z_min)/(z_max-z_min)       * 2 * np.pi 
    B[:,3] =  (B[:,3]-r_min)/(r_max-r_min)       * 2 * np.pi 
    B[:,4] =  (B[:,4]-phi_min)/(phi_max-phi_min) * 2 * np.pi 
    B[:,5] =  (B[:,5]-z_min)/(z_max-z_min)       * 2 * np.pi 
    return B
def preprocess(data):
    X,Ro,Ri,y = data
    X[:,2] = np.abs(X[:,2]) # correction for negative z
    bo    = np.dot(Ro.T, X)
    bi    = np.dot(Ri.T, X)
    B     = np.concatenate((bo,bi),axis=1)
    return map2angle(B), y
def binary_cross_entropy(output,label):
    return -(label*torch.log(output+1e-6) + (1-label)*torch.log(1-output+1e-6))
def cost(edge,theta_learn,y):
    out = qc(edge,theta_learn)
    return torch.mean(binary_cross_entropy(out,y))
    
input_dir   = 'data/hitgraphs_big'  
n_files     = 16*100
n_valid     = 1 # int(n_files * 0.1)
n_train     = 1 # n_files - n_valid 
train_data, valid_data = get_datasets(input_dir, n_train, n_valid)
B, y = preprocess(train_data[0])
num_epoch = 10
theta_learn = torch.tensor(np.random.rand(11)*np.pi*2,requires_grad=True)
loss_list = []
edge = torch.tensor(B)
label = torch.tensor(y)
qc = TorchCircuit.apply 
opt = torch.optim.Adam([theta_learn], lr=0.1)
for epoch in range(num_epoch):
    opt.zero_grad()
    loss = cost(edge[0:10,:],theta_learn,label[0:10])
    print(loss.item())
    loss.backward()
    opt.step()
    loss_list.append(loss.item())

plt.plot(loss_list)
#plt.show()




