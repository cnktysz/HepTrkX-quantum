import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics

def binary_cross_entropy(output,label):
	return -(label*np.log(output+1e-6) + (1-label)*np.log(1-output+1e-6))

output = np.array([2*(i-50)/100 for i in range(101)])
output = (output+1)/2
loss_0 = binary_cross_entropy(output,0)
loss_1 = binary_cross_entropy(output,1)

# Validation Plots
plt.clf()   
plt.plot(output,loss_0,label='label 0',c='darkorange')
plt.plot(output,loss_1,label='label 1',c='navy')
plt.xlabel('Output')
plt.ylabel('Loss')
plt.legend()
plt.savefig('png/binary_cross_entropy_loss.pdf')



