import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

log_location0 = 'logs/tensorflow/ENE/lr_0_01/'
log_location1 = 'logs/tensorflow/ENE2/lr_0_01/'
log_location2 = 'logs/tensorflow/ENE3/lr_0_01/'

png_location = 'png/comparison/lr_0_01/'


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
plt.plot(x0,valid0[:,2],label='iteration=1',c='darkorange')
plt.plot(x1,valid1[:,2],label='iteration=2',c='navy')
plt.plot(x2,valid2[:,2],label='iteration=3',c='red')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(png_location+'validation_loss.pdf')

plt.clf()   
plt.plot(x0,valid0[:,1],label='iteration=1',c='darkorange')
plt.plot(x1,valid1[:,1],label='iteration=2',c='navy')
plt.plot(x2,valid2[:,1],label='iteration=3',c='red')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.legend()
plt.savefig(png_location+'validation_auc.pdf')


x0 = [i+1 for i  in range(len(loss0))]
x1 = [i+1 for i  in range(len(loss1))]
x2 = [i+1 for i  in range(len(loss2))]

plt.clf()   
plt.plot(x0,loss0,label='iteration=1',c='darkorange')
plt.plot(x1,loss1,label='iteration=2',c='navy')
plt.plot(x2,loss2,label='iteration=3',c='red')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(png_location+'loss.pdf')

