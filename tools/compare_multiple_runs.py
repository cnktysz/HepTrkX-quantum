import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

log_loc0 = 'logs/TTN_hid1/'
#log_loc1 = 'logs/TTN_hid2/'
log_loc1 = 'logs/TTN_hid1_it1/'

#pdf_location = 'pdf/comparison/qgnn/dimension_comparison/'
#png_location = 'png/comparison/qgnn/dimension_comparison/'

pdf_location = 'pdf/comparison/qgnn/iteration_comparison/'
png_location = 'png/comparison/qgnn/iteration_comparison/'


n_runs0 = 5
n_runs1 = 3

log_list0 = [log_loc0 + 'run' + str(i+1) + '/' for i in range(n_runs0)]
log_list1 = [log_loc1 + 'run' + str(i+1) + '/' for i in range(n_runs1)]

print('Plots will be saved to:')
print('PDF: ' + pdf_location)
print('PNG: ' + png_location)

interval = 50
n_items0 = 29 
n_items1 = 12
# length of the arrays
accuracy0 = np.empty(shape=(n_runs0,n_items0))
auc0 = np.empty(shape=(n_runs0,n_items0))
loss0 = np.empty(shape=(n_runs0,n_items0))
precision0 = np.empty(shape=(n_runs0,n_items0))

accuracy1 = np.empty(shape=(n_runs1,n_items1))
auc1 = np.empty(shape=(n_runs1,n_items1))
loss1 = np.empty(shape=(n_runs1,n_items1))
precision1 = np.empty(shape=(n_runs1,n_items1))

for i in range(n_runs0):
	with open(log_list0[i]+'log_validation.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')  
		validation = np.array(list(reader)).astype(float)
		accuracy0[i,:] = validation[0:n_items0,0]	
		auc0[i,:] = validation[0:n_items0,1]
		loss0[i,:] = validation[0:n_items0,2]		
		precision0[i,:] = validation[0:n_items0,3]	

for i in range(n_runs1):
	with open(log_list1[i]+'log_validation.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')  
		validation = np.array(list(reader)).astype(float)
		accuracy1[i,:] = validation[0:n_items1,0]	
		auc1[i,:] = validation[0:n_items1,1]
		loss1[i,:] = validation[0:n_items1,2]		
		precision1[i,:] = validation[0:n_items1,3]	


x0 = [i*interval for i  in range(n_items0)]
x1 = [i*interval for i  in range(n_items1)]
# Plot Accuracy
plt.clf()   
plt.errorbar(x0,np.mean(accuracy0,axis=0),yerr=np.std(accuracy0,axis=0),c='navy')
plt.errorbar(x1,np.mean(accuracy1,axis=0),yerr=np.std(accuracy1,axis=0),c='darkorange')
plt.title('Validation Accuracy')
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig(pdf_location+'validation_accuracy.pdf')
plt.savefig(png_location+'validation_accuracy.png')
# Plot AUC
plt.clf()   
plt.errorbar(x0,np.mean(auc0,axis=0),yerr=np.std(auc0,axis=0),c='navy')
plt.errorbar(x1,np.mean(auc1,axis=0),yerr=np.std(auc1,axis=0),c='darkorange')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.tight_layout()
plt.savefig(pdf_location+'validation_auc.pdf')
plt.savefig(png_location+'validation_auc.png')
#Plot Precision
plt.clf()   
plt.errorbar(x0,np.mean(precision0,axis=0),yerr=np.std(precision0,axis=0),c='navy')
plt.errorbar(x1,np.mean(precision1,axis=0),yerr=np.std(precision1,axis=0),c='darkorange')
plt.title('Validation Precision')
plt.xlabel('Update')
plt.ylabel('Precision')
plt.tight_layout()
plt.savefig(pdf_location+'validation_precision.pdf')
plt.savefig(png_location+'validation_precision.png')
# Plot Loss
plt.clf()   
plt.errorbar(x0,np.mean(loss0,axis=0),yerr=np.std(loss0,axis=0),c='navy')
plt.errorbar(x1,np.mean(loss1,axis=0),yerr=np.std(loss1,axis=0),c='darkorange')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(pdf_location+'validation_loss.pdf')
plt.savefig(png_location+'validation_loss.png')


loss0 = np.empty(shape=(n_runs0,(n_items0-1)*interval))
loss1 = np.empty(shape=(n_runs1,(n_items1-1)*interval))
for i in range(n_runs0):
	with open(log_list0[i]+'log_loss.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		training = np.array(list(reader)).astype(float)
		loss0[i,:] = training[0:((n_items0-1)*interval),0]	
for i in range(n_runs1):
	with open(log_list1[i]+'log_loss.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		training = np.array(list(reader)).astype(float)
		loss1[i,:] = training[0:((n_items1-1)*interval),0]	

x0 = [i for i  in range((n_items0-1)*interval)]
x1 = [i for i  in range((n_items1-1)*interval)]

plt.clf()   
plt.errorbar(x0,np.mean(loss0,axis=0),yerr=0,c='navy')
plt.errorbar(x1,np.mean(loss1,axis=0),yerr=0,c='darkorange')
plt.title('Mean of Training Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(pdf_location+'training_loss.pdf')
plt.savefig(png_location+'training_loss.png')

