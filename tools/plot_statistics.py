import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

log_location = 'logs/tensorflow/ENE/lr_0_1/'
png_location = 'png/tensorflow/ENE/lr_0_1/'

with open(log_location + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss = np.array(list(reader)).astype(float)
with open(log_location+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid = np.array(list(reader)).astype(float) 
with open(log_location+'log_learning_variables.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	learning_vars = np.array(list(reader))[:,0:-1].astype(float)
with open(log_location+'log_grads.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	grads = np.array(list(reader))[:,0:-1].astype(float)
with open(log_location + 'log_validation_preds.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid_preds = np.array(list(reader)).astype(float)

# Plots
plt.clf()   
x = [(i+1) for i  in range(len(loss))]
plt.plot(x[1:],loss[1:])
plt.xlabel('Update')
plt.ylabel('Loss')
plt.savefig(png_location+'statistics_loss.pdf')

# Validation Plots
plt.clf()   
interval = 50
x = [i*interval for i  in range(len(valid))]
plt.plot(x,valid[:,2])
plt.xlabel('Update')
plt.ylabel('Loss')
plt.savefig(png_location+'validation_loss.pdf')

plt.clf()   
x = [i*interval for i  in range(len(valid))]
plt.plot(x,valid[:,0]*100)
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.savefig(png_location+'validation_accuracy.pdf')

plt.clf()   
x = [i*interval for i  in range(len(valid))]
plt.plot(x,valid[:,1])
plt.xlabel('Update')
plt.ylabel('AUC')
plt.savefig(png_location+'validation_auc.pdf')


fpr,tpr,thresholds = metrics.roc_curve(valid_preds[:,1].astype(int),valid_preds[:,0],pos_label=1 )
auc = metrics.auc(fpr,tpr)
# Plot
plt.clf()   
plt.plot(fpr,tpr,c='navy')
plt.plot([0, 1], [0, 1], color='darkorange', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(png_location+'validation_roc.pdf') 


fig, axs = plt.subplots(2,2)
fig.suptitle('Learning Variables')
x = [i for i  in range(len(learning_vars))]
for i in range(0,3):
	axs[0,0].scatter(x,learning_vars[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
for i in range(3,18):
	axs[0,1].scatter(x,learning_vars[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
for i in range(18,41):
	axs[1,0].scatter(x,learning_vars[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
for i in range(41,56):
	axs[1,1].scatter(x,learning_vars[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
fig.savefig(png_location+'statistics_learning_variables.pdf')

fig, axs = plt.subplots(2,2)
fig.suptitle('Learning Variable Gradients')
x = [(i+1) for i  in range(len(grads))]
for i in range(0,3):
	axs[0,0].scatter(x,grads[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
for i in range(3,18):
	axs[0,1].scatter(x,grads[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
for i in range(18,41):
	axs[1,0].scatter(x,grads[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
for i in range(41,56):
	axs[1,1].scatter(x,grads[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
fig.savefig(png_location+'statistics_grads.pdf')


