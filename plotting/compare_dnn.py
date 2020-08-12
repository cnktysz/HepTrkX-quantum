import matplotlib.pyplot as plt 
import numpy as np
import csv, sys

log_location0 = 'logs/comparisons/dnn/hid1/'
log_location1 = 'logs/comparisons/dnn/hid5/'
log_location2 = 'logs/comparisons/dnn/hid10/'
log_location3 = 'logs/comparisons/dnn/hid20/'
png_location = 'png/comparison/dnn/'


with open(log_location0+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid0 = np.array(list(reader)).astype(float) 
with open(log_location1+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid1 = np.array(list(reader)).astype(float) 
with open(log_location2+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid2 = np.array(list(reader)).astype(float) 
with open(log_location3+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid3 = np.array(list(reader)).astype(float) 

with open(log_location0 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss0 = np.array(list(reader)).astype(float)
with open(log_location1 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss1 = np.array(list(reader)).astype(float)
with open(log_location2 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss2 = np.array(list(reader)).astype(float)
with open(log_location3 + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss3 = np.array(list(reader)).astype(float)
'''
with open(log_location0+'log_training.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	train0 = np.array(list(reader)).astype(float) 
with open(log_location1+'log_training.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	train1 = np.array(list(reader)).astype(float) 
with open(log_location2+'log_training.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	train2 = np.array(list(reader)).astype(float) 
with open(log_location3+'log_training.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	train3 = np.array(list(reader)).astype(float) 
'''
interval = 200 
x0 = [i*interval for i  in range(len(valid0))]
x1 = [i*interval for i  in range(len(valid1))]
x2 = [i*interval for i  in range(len(valid2))]
x3 = [i*interval for i  in range(len(valid3))]

plt.clf()   
plt.plot(x0,valid0[:,2],label=r'$N_{hid}=1$',c='darkorange')
plt.plot(x1,valid1[:,2],label=r'$N_{hid}=5$',c='navy')
plt.plot(x2,valid2[:,2],label=r'$N_{hid}=10$',c='red')
plt.plot(x3,valid3[:,2],label=r'$N_{hid}=20$',c='green')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(png_location+'validation_loss.pdf')
'''
plt.clf()   
plt.plot(x0,train0[:,2],label=r'$N_{hid}=1$',c='darkorange')
plt.plot(x1,train1[:,2],label=r'$N_{hid}=5$',c='navy')
plt.plot(x2,train2[:,2],label=r'$N_{hid}=10$',c='red')
plt.plot(x3,train3[:,2],label=r'$N_{hid}=20$',c='green')
plt.title('Training Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(png_location+'training_loss.pdf')
'''
plt.clf()   
plt.plot(x0,valid0[:,1],label=r'$N_{hid}=1$',c='darkorange')
plt.plot(x1,valid1[:,1],label=r'$N_{hid}=5$',c='navy')
plt.plot(x2,valid2[:,1],label=r'$N_{hid}=10$',c='red')
plt.plot(x3,valid3[:,1],label=r'$N_{hid}=20$',c='green')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.legend()
plt.savefig(png_location+'validation_auc.pdf')
'''
plt.clf()   
plt.plot(x0,train0[:,1],label=r'$N_{hid}=1$',c='darkorange')
plt.plot(x1,train1[:,1],label=r'$N_{hid}=5$',c='navy')
plt.plot(x2,train2[:,1],label=r'$N_{hid}=10$',c='red')
plt.plot(x3,train3[:,1],label=r'$N_{hid}=20$',c='green')
plt.title('Training AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.legend()
plt.savefig(png_location+'training_auc.pdf')
'''
x0 = [i+1 for i  in range(len(loss0))]
x1 = [i+1 for i  in range(len(loss1))]
x2 = [i+1 for i  in range(len(loss2))]
x3 = [i+1 for i  in range(len(loss3))]

plt.clf()   
plt.plot(x0,loss0,label=r'$N_{hid}=1$',c='darkorange')
plt.plot(x1,loss1,label=r'$N_{hid}=5$',c='navy')
plt.plot(x2,loss2,label=r'$N_{hid}=10$',c='red')
plt.plot(x3,loss3,label=r'$N_{hid}=20$',c='green')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.savefig(png_location+'loss.pdf')

