import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

#log_loc = 'logs/comparisons/dnn/dim-comparison/'
log_loc = 'logs/comparisons/qgnn/iteration_comparison/'

#log_location0 = log_loc + 'hid1/'
#log_location1 = log_loc + 'hid10/'
#log_location2 = log_loc + 'hid100/'

log_location0 = log_loc + 'it1/'
log_location1 = log_loc + 'it2/'
log_location2 = log_loc + 'it3/'

pdf_location = 'pdf/comparison/qgnn/iteration_comparison/'

#label1 = r'$N_{dim}=1$'
#label2 = r'$N_{dim}=10$'
#label3 = r'$N_{dim}=100$'

label1 = r'$N_{it}=1$'
label2 = r'$N_{it}=2$'
label3 = r'$N_{it}=3$'

params0 = [['']]
params1 = [['']]
params2 = np.zeros([1,41])

with open(log_location2+'log_params.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	list2 = list(reader)
	for i in range(len(list2)):
		params2 = np.append(params2,list2[i],axis=0)

	print(params2.shape)


with open(log_location0+'log_params.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	list0 = np.array(list(reader))


with open(log_location1+'log_params.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	list1 = np.array(list(reader))
	for i in range(len(list2)):
		params1 = np.append(params1, np.array(list1[i])[0:-1].astype(float),axis=0)


print(params0)

x = [(i+1) for i  in range(len(params2))]


fig, axs = plt.subplots(3,3,sharey=True)
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
fig.savefig(pdf_location+'params.pdf')




