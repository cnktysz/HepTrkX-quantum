import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

log_location0 = 'logs/tensorflow/ENE/lr_0_1/'
log_location1 = 'logs/tensorflow/TTN/lr_0_01/'
png_location = 'png/comparison/'


with open(log_location0+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid0 = np.array(list(reader)).astype(float) 
with open(log_location1+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid1 = np.array(list(reader)).astype(float) 


plt.clf()   
interval = 50
x0 = [i*interval for i  in range(len(valid0))]
x1 = [i*interval for i  in range(len(valid1))]
plt.plot(x0,valid0[:,2],label='E->N->E',c='darkorange')
plt.plot(x1,valid1[:,2],label='E',c='navy')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(png_location+'validation_loss.pdf')

plt.clf()   
plt.plot(x0,valid0[:,1],label='E->N->E',c='darkorange')
plt.plot(x1,valid1[:,1],label='E',c='navy')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.legend()
plt.savefig(png_location+'validation_auc.pdf')



