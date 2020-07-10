import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

log_loc0 = 'logs/comparisons/dnn/dim-comparison/'
log_loc1 = 'logs/comparisons/qgnn/iteration_comparison/'
log_loc2 = 'logs/comparisons/qgnn/learning_rate_comparison/'
log_loc3 = 'logs/comparisons/qgnn/dimension_comparison/'
log_loc4 = 'logs/comparisons/qgnn/arch_comparison/'

run_type = 'test'

# TO DO:
# some run types are deprecated, need to update them

# TO DO:
# implement exectuable input form
# ask run_type as input

if run_type == 'cla-qua':
	log_location0 = log_loc0 + 'hid1/'
	log_location1 = log_loc0 + 'hid10/'
	log_location2 = log_loc0 + 'hid100/'
	log_location3 = log_loc1 + 'it1/'

	pdf_location = 'pdf/comparison/cla-qua/'
	png_location = 'png/comparison/cla-qua/'

	label0 = r'$Classical \, Network \, with \, N_{Dim} = 1$'
	label1 = r'$Classical \, Network \, with \, N_{Dim} = 10$'
	label2 = r'$Classical \, Network \, with \, N_{Dim} = 100$'
	label3 = r'$Quantum \, Network \, with \, N_{Dim} = 1$'


if run_type == 'classical_dimension_comparison':
	log_location0 = log_loc + 'hid1/'
	log_location1 = log_loc + 'hid10/'
	log_location2 = log_loc + 'hid100/'

	label1 = r'$N_{dim}=1$'
	label2 = r'$N_{dim}=10$'
	label3 = r'$N_{dim}=100$'


if run_type == 'iteration_comparison':
	# Iteration comparison
	n_runs = 4
	log_location0 = log_loc1 + 'it1/'
	log_location1 = log_loc1 + 'it2/'
	log_location2 = log_loc1 + 'it3/'
	log_location3 = log_loc1 + 'it4/'

	pdf_location = 'pdf/comparison/qgnn/iteration_comparison/'
	png_location = 'png/comparison/qgnn/iteration_comparison/'

	label0 = r'$N_{it}=1$'
	label1 = r'$N_{it}=2$'
	label2 = r'$N_{it}=3$'
	label3 = r'$N_{it}=4$'

if run_type == 'l.rate_comparison':
	# Learning rate comparison
	n_runs = 3
	log_location0 = log_loc2 + 'lr_1e-2/'
	log_location1 = log_loc2 + 'lr_2e-2/'
	log_location2 = log_loc2 + 'lr_3e-2/'

	pdf_location = 'pdf/comparison/qgnn/learning_rate_comparison/'
	png_location = 'png/comparison/qgnn/learning_rate_comparison/'

	label0 = r'$lr=1x10^{-2}$'
	label1 = r'$lr=2x10^{-2}$'
	label2 = r'$lr=3x10^{-2}$'

if run_type == 'dimension_comparison':
	## Dimension comparison
	n_runs = 2
	log_location0 = log_loc2 + 'lr_3e-2/'
	log_location1 = log_loc3 + 'dim2/'

	pdf_location = 'pdf/comparison/qgnn/dimension_comparison/'
	png_location = 'png/comparison/qgnn/dimension_comparison/'

	label0 = r'$N_{dim}=1$'
	label1 = r'$N_{dim}=2$'

if run_type == 'test':## Test comparison
	n_runs = 2
	log_location0 = log_loc2 + 'lr_3e-2/'
	log_location1 = 'logs/test/'


	pdf_location = 'pdf/test/'
	png_location = 'png/test/'

	label0 = 'base'
	label1 = 'test'


if run_type == 'arch_comparison':
	## General vs. Real Rotations
	n_runs = 2
	log_location0 = log_loc2 + 'lr_3e-2/'
	log_location1 = log_loc4 + 'embedding_learning/'

	pdf_location = 'pdf/comparison/qgnn/arch_comparison/'
	png_location = 'png/comparison/qgnn/arch_comparison/'

	label0 = 'Old Architecture'
	label1 = 'New(test) Architecture'


print('Comparison plots will be saved to:')
print('PDF: ' + pdf_location)
print('PNG: ' + png_location)
print('Comparing %d runs................'%n_runs)

interval = 50

with open(log_location0+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid0 = np.array(list(reader)).astype(float)
	valid0 = valid0[:28] 
x0 = [i*interval for i  in range(len(valid0))]
with open(log_location1+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid1 = np.array(list(reader)).astype(float) 
	valid1 = valid1[:28] 
x1 = [i*interval for i  in range(len(valid1))]
if n_runs > 2:
	with open(log_location2+'log_validation.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		valid2 = np.array(list(reader)).astype(float) 
		valid2 = valid2[:28] 
	x2 = [i*interval for i  in range(len(valid2))]
if n_runs > 3 :
	with open(log_location3+'log_validation.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		valid3 = np.array(list(reader)).astype(float) 
		valid3 = valid3[:28] 
	x3 = [i*interval for i  in range(len(valid3))]


plt.clf()   
plt.plot(x0,valid0[:,0],label=label0,c='darkorange')
plt.plot(x1,valid1[:,0],label=label1,c='navy')
if n_runs > 2: plt.plot(x2,valid2[:,0],label=label2,c='red')
if n_runs > 3: plt.plot(x3,valid3[:,0],label=label3,c='green')
plt.title('Validation Accuracy')
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(pdf_location+'validation_accuracy.pdf')
plt.savefig(png_location+'validation_accuracy.png')


plt.clf()   
plt.plot(x0,valid0[:,1],label=label0,c='darkorange')
plt.plot(x1,valid1[:,1],label=label1,c='navy')
if n_runs > 2: plt.plot(x2,valid2[:,1],label=label2,c='red')
if n_runs > 3: plt.plot(x3,valid3[:,1],label=label3,c='green')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.legend()
plt.savefig(pdf_location+'validation_auc.pdf')
plt.savefig(png_location+'validation_auc.png')

plt.clf()   
plt.plot(x0,valid0[:,2],label=label0,c='darkorange')
plt.plot(x1,valid1[:,2],label=label1,c='navy')
if n_runs > 2: plt.plot(x2,valid2[:,2],label=label2,c='red')
if n_runs > 3: plt.plot(x3,valid3[:,2],label=label3,c='green')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(pdf_location+'validation_loss.pdf')
plt.savefig(png_location+'validation_loss.png')


with open(log_location0 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss0 = np.array(list(reader)).astype(float)
	loss0 = loss0[:1399] 
x0 = [i+1 for i  in range(len(loss0))]

with open(log_location1 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss1 = np.array(list(reader)).astype(float)
	loss1 = loss1[:1399] 
x1 = [i+1 for i  in range(len(loss1))]

if n_runs > 2:
	with open(log_location2 + 'log_loss.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		loss2 = np.array(list(reader)).astype(float)
		loss2 = loss2[:1399] 
	x2 = [i+1 for i  in range(len(loss2))]

if n_runs > 3:
	with open(log_location3 + 'log_loss.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		loss3 = np.array(list(reader)).astype(float)
		loss3 = loss3[:1399] 
	x3 = [i+1 for i  in range(len(loss3))]


plt.clf()   
plt.plot(x0,loss0,label=label0,c='darkorange')
plt.plot(x1,loss1,label=label1,c='navy')
if n_runs > 2: plt.plot(x2,loss2,label=label2,c='red')
if n_runs > 3: plt.plot(x3,loss3,label=label3,c='green')
plt.title('Training Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(pdf_location+'training_loss.pdf')
plt.savefig(png_location+'training_loss.png')

