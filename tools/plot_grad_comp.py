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

with open(log_location0+'log_grads.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	grads0 = np.array(list(reader))[:,0:-1].astype(float)
with open(log_location1+'log_grads.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	grads1 = np.array(list(reader))[:,0:-1].astype(float)
with open(log_location2+'log_grads.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	grads2 = np.array(list(reader))[:,0:-1].astype(float) 


x = [(i+1) for i  in range(len(grads2))]
grads0 = grads0[:len(grads2),:]
grads1 = grads1[:len(grads2),:]
'''
fig, axs = plt.subplots(3,3,sharey=True)
for i in range(3):
	axs[0,0].scatter(x,grads0[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[0,0].set_title('InputNet')
for i in range(3,18):
	axs[0,1].scatter(x,grads0[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[0,1].set_title('QuantumEdgeNet')
for i in range(18,41):
	axs[0,2].scatter(x,grads0[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[0,2].set_title('QuantumNodeNet')
for i in range(3):
	axs[1,0].scatter(x,grads1[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[1,0].set_title('InputNet')
for i in range(3,18):
	axs[1,1].scatter(x,grads1[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[1,1].set_title('QuantumEdgeNet')
for i in range(18,41):
	axs[1,2].scatter(x,grads1[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[1,2].set_title('QuantumNodeNet')
for i in range(3):
	axs[2,0].scatter(x,grads2[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[2,0].set_title('InputNet')
for i in range(3,18):
	axs[2,1].scatter(x,grads2[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[2,1].set_title('QuantumEdgeNet')
for i in range(18,41):
	axs[2,2].scatter(x,grads2[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
	axs[2,2].set_title('QuantumNodeNet')
fig.tight_layout()
fig.savefig(pdf_location+'grads.pdf')

avg_grads0 = np.mean(grads0,axis=1)
avg_grads1 = np.mean(grads1,axis=1)
avg_grads2 = np.mean(grads2,axis=1)

avg0 = np.mean(np.abs(avg_grads0))
avg1 = np.mean(np.abs(avg_grads1))
avg2 = np.mean(np.abs(avg_grads2))

plt.clf()   
plt.plot(x,avg_grads0,label=label1+r' Total avg: %.6f'%avg0 ,c='darkorange')
plt.plot(x,avg_grads1,label=label2+r' Total avg: %.6f'%avg1 ,c='navy')
plt.plot(x,avg_grads2,label=label3+r' Total avg: %.6f'%avg2 ,c='red')
plt.title('Average Gradients')
plt.xlabel('Update')
plt.ylabel('Avg')
plt.legend()
plt.savefig(pdf_location+'grad_avg.pdf')
'''


fig, axs = plt.subplots(3,1,sharex=True, figsize=(5,20))
fig.tight_layout()

plt.xlabel('Update')
plt.xlabel('Gradient')
axs[0].set_title('InputNet')
for i in range(3):
	axs[i].plot(x,grads0[:,i],label=label1)
	axs[i].plot(x,grads1[:,i],label=label2)
	axs[i].plot(x,grads2[:,i],label=label3)

plt.legend()
fig.savefig(pdf_location+'grads_InputNet.pdf')

plt.clf()  
fig, axs = plt.subplots(15,1,sharex=True,figsize=(5,20))


plt.xlabel('Update')
plt.xlabel('Gradient')
axs[0].set_title('EdgeNet')
for i in range(3,18):
	axs[i-3].plot(x,grads0[:,i],label=label1)
	axs[i-3].plot(x,grads1[:,i],label=label2)
	axs[i-3].plot(x,grads2[:,i],label=label3)
	
plt.legend()
fig.tight_layout()
fig.savefig(pdf_location+'grads_EdgeNet.pdf')

plt.clf()  
fig, axs = plt.subplots(23,1,sharex=True,figsize=(5,20))


plt.xlabel('Update')
plt.xlabel('Gradient')
axs[0].set_title('NodeNet')
for i in range(18,41):
	axs[i-18].plot(x,grads0[:,i],label=label1)
	axs[i-18].plot(x,grads1[:,i],label=label2)
	axs[i-18].plot(x,grads2[:,i],label=label3)

plt.legend()
fig.tight_layout()
fig.savefig(pdf_location+'grads_NodeNet.pdf')