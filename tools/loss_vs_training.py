import matplotlib.pyplot as plt 
import numpy as np
import csv, sys

log_location = 'logs/comparisons/dnn/hid20/'

png_location = 'png/dnn/hid20/'


with open(log_location+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid = np.array(list(reader)).astype(float) 

with open(log_location+'log_training.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	train = np.array(list(reader)).astype(float) 


interval = 200 
x = [i*interval for i  in range(len(valid))]

plt.clf()   
plt.plot(x,valid[:,2],label='validation',c='darkorange')
plt.plot(x,train[:,2],label='training',c='navy')
plt.title('Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(png_location+'loss_comparison.pdf')

plt.clf()   
plt.plot(x,valid[:,1],label='validaiton',c='darkorange')
plt.plot(x,train[:,1],label='training',c='navy')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.legend()
plt.savefig(png_location+'auc_comparison.pdf')



