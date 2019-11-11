import sys
import numpy as np
from qiskit import *
from qiskit.providers.aer import noise

q       = QuantumRegister(1)
c       = ClassicalRegister(1)
circuit = QuantumCircuit(q,c)

#IBMQ.save_account('64c6b5846661587a531474eb03a1277400bd8f79bdf3131bb4e834370f7ca24c8c85759573c25dfef8046ba1cb4479363b0735cb168d35fb6721ed081b7bf25a')
provider = IBMQ.load_account()
#provider = IBMQ.providers()
backends = provider.backends()
device = provider.get_backend('ibmq_16_melbourne')
properties = device.properties()

#ibmqx4 = IBMQ.get_backend('ibmq_16_melbourne')
#device_properties = ibmqx4.properties()

noise_model = noise.device.basic_device_noise_model(properties)

circuit.x(q)

circuit.measure(q,c)

shots=1000
backend = Aer.get_backend('qasm_simulator')
result = qiskit.execute(circuit, backend, shots=shots,
                                   noise_model=noise_model).result()
print(result.get_counts(circuit))