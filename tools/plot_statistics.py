import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
from scipy import special
import sys

def plot_statistics(log_location,png_location):
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

	plt.figure(num=None, figsize=None)
	# Plots
	plt.clf()   
	x = [(i+1) for i  in range(len(loss))]
	plt.plot(x[1:],loss[1:])
	plt.xlabel('Update')
	plt.ylabel('Loss')
	plt.tight_layout()
	plt.savefig(png_location+'statistics_loss.pdf')

	# Validation Plots
	plt.clf()   
	interval = 50
	x = [i*interval for i  in range(len(valid))]
	plt.plot(x,valid[:,2])
	plt.xlabel('Update')
	plt.ylabel('Loss')
	plt.tight_layout()
	plt.savefig(png_location+'validation_loss.pdf')

	plt.clf()   
	x = [i*interval for i  in range(len(valid))]
	plt.plot(x,valid[:,0]*100)
	plt.xlabel('Update')
	plt.ylabel('Accuracy')
	plt.tight_layout()
	plt.savefig(png_location+'validation_accuracy.pdf')

	plt.clf()   
	x = [i*interval for i  in range(len(valid))]
	plt.plot(x,valid[:,1])
	plt.xlabel('Update')
	plt.ylabel('AUC')
	plt.tight_layout()
	plt.savefig(png_location+'validation_auc.pdf')
	
	y_pred = valid_preds[:,0]
	y_true = valid_preds[:,1].astype(int)
	fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred,pos_label=1)
	auc = metrics.auc(fpr,tpr)
	print(auc)
	# Plot
	plt.clf()   
	plt.plot(fpr,tpr,c='navy')
	plt.plot([0, 1], [0, 1], color='darkorange', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.tight_layout()
	plt.savefig(png_location+'validation_roc.pdf') 

	# Plot the model outputs
	
	plt.clf()
	binning=dict(bins=50, range=(0,1), histtype='bar', log=True)
	plt.hist(valid_preds[valid_preds[:,1]==0,0], label='fake', **binning)
	plt.hist(valid_preds[valid_preds[:,1]==1,0], label='true', **binning)
	plt.xlabel('Model output')
	plt.legend(loc=0)
	plt.savefig(png_location+'validation_outputs.pdf') 


	fig, axs = plt.subplots(1,3,figsize=(10,4))
	x = [i for i  in range(len(learning_vars))]
	for i in range(3):
		axs[0].scatter(x,learning_vars[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
		axs[0].set_title('InputNet')
		axs[0].set_xlabel('Update')
	for i in range(3,18):
		axs[1].scatter(x,learning_vars[:,i]%(np.pi*2),marker='.',label=r'$\theta_{'+str(i)+'}$')
		axs[1].set_title('QuantumEdgeNet')
		axs[1].set_ylabel('Angle')
		axs[1].set_xlabel('Update')
	for i in range(18,41):
		axs[2].scatter(x,learning_vars[:,i]%(np.pi*2),marker='.',label=r'$\theta_{'+str(i)+'}$')
		axs[2].set_title('QuantumNodeNet')
		axs[2].set_ylabel('Angle')
		axs[2].set_xlabel('Update')
	fig.tight_layout()
	fig.savefig(png_location+'statistics_learning_variables.pdf')

	fig, axs = plt.subplots(1,3,figsize=(10,4))
	x = [(i+1) for i  in range(len(grads))]
	for i in range(3):
		axs[0].scatter(x,grads[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
		axs[0].set_title('InputNet')
		axs[0].set_xlabel('Update')
	for i in range(3,18):
		axs[1].scatter(x,grads[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
		axs[1].set_title('QuantumEdgeNet')
		axs[1].set_ylabel('Angle')
		axs[1].set_xlabel('Update')
	for i in range(18,41):
		axs[2].scatter(x,grads[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
		axs[2].set_title('QuantumNodeNet')
		axs[2].set_ylabel('Angle')
		axs[2].set_xlabel('Update')
	fig.tight_layout()
	fig.savefig(png_location+'statistics_grads.pdf')
########################################################
file_list = ['ENE/lr_0_01/','ENE2/lr_0_01/','ENE3/lr_0_01/','ENE/lr_0_1/','ENE2/lr_0_1/','ENE3/lr_0_1/','ENE8/lr_0_1/']

for i in range(len(file_list)):
	file_name = file_list[i]
	log_location = 'logs/tensorflow/' + file_name
	png_location = 'png/tensorflow/' + file_name
	plot_statistics(log_location,png_location)


