import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

#log_loc = 'logs/comparisons/dnn/dim-comparison/'
log_loc = 'logs/comparisons/qgnn/params/'

#log_location0 = log_loc + 'hid1/'
#log_location1 = log_loc + 'hid10/'
#log_location2 = log_loc + 'hid100/'

log_location0 = log_loc  
log_location1 = log_loc 
log_location2 = log_loc 

pdf_location = 'pdf/comparison/qgnn/iteration_comparison/'

#label1 = r'$N_{dim}=1$'
#label2 = r'$N_{dim}=10$'
#label3 = r'$N_{dim}=100$'

label1 = r'$N_{it}=1$'
label2 = r'$N_{it}=2$'
label3 = r'$N_{it}=3$'

with open(log_location0+'params1.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	params0 = np.array(list(reader))[:,0:-1].astype(float)
with open(log_location1+'params2.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	params1 = np.array(list(reader))[:,0:-1].astype(float)
with open(log_location2+'params3.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	params2 = np.array(list(reader))[:,0:-1].astype(float) 


fig, axs = plt.subplots(3,3,sharey=True,figsize=(20,10))
x = [(i+1) for i  in range(len(params2))]
params0 = params0[:len(params2),:]
params1 = params1[:len(params2),:]
for i in range(3):
	axs[0,0].scatter(x,params0[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[0,0].set_title('InputNet')
for i in range(3,18):
	axs[0,1].scatter(x,params0[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[0,1].set_title('QuantumEdgeNet')
for i in range(18,41):
	axs[0,2].scatter(x,params0[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[0,2].set_title('QuantumNodeNet')
for i in range(3):
	axs[1,0].scatter(x,params1[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[1,0].set_title('InputNet')
for i in range(3,18):
	axs[1,1].scatter(x,params1[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[1,1].set_title('QuantumEdgeNet')
for i in range(18,41):
	axs[1,2].scatter(x,params1[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[1,2].set_title('QuantumNodeNet')
for i in range(3):
	axs[2,0].scatter(x,params2[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[2,0].set_title('InputNet')
for i in range(3,18):
	axs[2,1].scatter(x,params2[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[2,1].set_title('QuantumEdgeNet')
for i in range(18,41):
	axs[2,2].scatter(x,params2[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[2,2].set_title('QuantumNodeNet')
fig.tight_layout()
plt.show()
fig.savefig(pdf_location+'params.pdf')


