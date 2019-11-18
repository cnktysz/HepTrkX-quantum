import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics

log_location = 'logs/pennylane/TTN/lr_0_1/'
png_location = 'png/pennylane/TTN/lr_0_1/'
circuit_type = 'TTN'
if circuit_type=='TTN':
	n_param = 11
elif circuit_type=='MERA':
	n_param = 19
with open(log_location + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss = np.array(list(reader)).astype(float)
with open(log_location+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid = np.array(list(reader)).astype(float) 
with open(log_location+'log_theta.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	theta = np.delete(np.array(list(reader)[:-2]),n_param,1).astype(float)
with open(log_location + 'log_validation_preds.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid_preds = np.array(list(reader)).astype(float)
"""
with open(log_location+'log_training.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	train = np.array(list(reader)).astype(float) 
with open(log_location + 'log_training_preds.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	train_preds = np.array(list(reader)).astype(float)
"""
# Plots
plt.clf()   
x = [(i+1) for i  in range(len(loss))]
plt.plot(x[1:],loss[1:])
plt.xlabel('Update')
plt.ylabel('Loss')
plt.savefig(png_location+'statistics_loss.pdf')

plt.clf()
x = [(i+1) for i in range(len(theta))]
for i in range(n_param):
	plt.scatter(x,theta[:,i],marker='.',label=r'$\theta_{'+str(i)+'}$')
plt.xlabel('Update')
plt.ylabel(r'Angle (0 - 2$\pi$)')
plt.savefig(png_location+'statistics_angle.pdf')

# Validation Plots
plt.clf()   
interval = 50
x = [i*interval for i  in range(len(valid))]
plt.plot(x,valid[:,2])
plt.xlabel('Update')
plt.ylabel('Loss')
plt.savefig(png_location+'validation_loss.pdf')

plt.clf()   
x = [i*50 for i  in range(len(valid))]
plt.plot(x,valid[:,0]*100)
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.savefig(png_location+'validation_accuracy.pdf')

plt.clf()   
x = [i*50 for i  in range(len(valid))]
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
"""
# Training Plots
plt.clf()   
interval = 50
x = [i*interval for i  in range(len(train))]
plt.plot(x,valid[:,2])
plt.xlabel('Update')
plt.ylabel('Loss')
plt.savefig(png_location+'training_loss.pdf')

plt.clf()   
x = [i*50 for i  in range(len(train))]
plt.plot(x,valid[:,0]*100)
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.savefig(png_location+'training_accuracy.pdf')

plt.clf()   
x = [i*50 for i  in range(len(train))]
plt.plot(x,valid[:,1])
plt.xlabel('Update')
plt.ylabel('AUC')
plt.savefig(png_location+'training_auc.pdf')

fpr,tpr,thresholds = metrics.roc_curve(train_preds[:,1].astype(int),train_preds[:,0],pos_label=1 )
auc = metrics.auc(fpr,tpr)
# Plot
plt.clf()   
plt.plot(fpr,tpr,c='navy')
plt.plot([0, 1], [0, 1], color='darkorange', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(png_location+'training_roc.pdf') 
"""


