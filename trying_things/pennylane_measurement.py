import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import matplotlib.pyplot as plt
import pennylane as qml 
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=6)

@qml.qnode(dev)
def circuit(variables):

	qml.RY(variables[0],wires=0)
	qml.RY(variables[1],wires=1)
	qml.CNOT(wires=[0,1])
	qml.RY(variables[2],wires=2)
	qml.RY(variables[3],wires=3)
	qml.CNOT(wires=[2,3])
	qml.RY(variables[4],wires=4)
	qml.RY(variables[5],wires=5)
	qml.CNOT(wires=[5,4])
	qml.RY(variables[6],wires=1)
	qml.RY(variables[7],wires=3)
	qml.CNOT(wires=[1,3])
	qml.RY(variables[8],wires=3)
	qml.RY(variables[9],wires=4)
	qml.CNOT(wires=[3,4])
	qml.RY(variables[10],wires=4)
	return qml.expval(qml.PauliZ(wires=4)), qml.expval(qml.PauliZ(wires=5))


variables = np.random.rand(11)

print(circuit(variables))