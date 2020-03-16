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


with open(log_location0+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid0 = np.array(list(reader)).astype(float) 
with open(log_location1+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid1 = np.array(list(reader)).astype(float) 
with open(log_location2+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid2 = np.array(list(reader)).astype(float) 
with open(log_location0 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss0 = np.array(list(reader)).astype(float)
with open(log_location1 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss1 = np.array(list(reader)).astype(float)
with open(log_location2 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss2 = np.array(list(reader)).astype(float)

interval = 50

x0 = [i*interval for i  in range(len(valid0))]
x1 = [i*interval for i  in range(len(valid1))]
x2 = [i*interval for i  in range(len(valid2))]

plt.clf()   
plt.plot(x0,valid0[:,0],label=label1,c='darkorange')
plt.plot(x1,valid1[:,0],label=label2,c='navy')
plt.plot(x2,valid2[:,0],label=label3,c='red')
plt.title('Validation Accuracy')
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(pdf_location+'validation_accuracy.pdf')

plt.clf()   
plt.plot(x0,valid0[:,1],label=label1,c='darkorange')
plt.plot(x1,valid1[:,1],label=label2,c='navy')
plt.plot(x2,valid2[:,1],label=label3,c='red')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.legend()
plt.savefig(pdf_location+'validation_auc.pdf')

plt.clf()   
plt.plot(x0,valid0[:,2],label=label1,c='darkorange')
plt.plot(x1,valid1[:,2],label=label2,c='navy')
plt.plot(x2,valid2[:,2],label=label3,c='red')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(pdf_location+'validation_loss.pdf')

x0 = [i+1 for i  in range(len(loss0))]
x1 = [i+1 for i  in range(len(loss1))]
x2 = [i+1 for i  in range(len(loss2))]

plt.clf()   
plt.plot(x0,loss0,label=label1,c='darkorange')
plt.plot(x1,loss1,label=label2,c='navy')
plt.plot(x2,loss2,label=label3,c='red')
plt.title('Training Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(pdf_location+'training_loss.pdf')

