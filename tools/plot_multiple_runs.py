import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

log_loc = 'logs/TTN_hid2/'

n_runs = 3

log_list = [log_loc + 'run' + str(i+1) + '/' for i in range(n_runs)]

pdf_location = 'pdf/TTN_hid2/'
png_location = 'png/TTN_hid2/'

print('Plots will be saved to:')
print('PDF: ' + pdf_location)
print('PNG: ' + png_location)
print('Plotting %d runs................'%n_runs)

interval = 50
n_items = 11 # need a better code organization to get rid of this parameter
# length of the arrays
accuracy = np.empty(shape=(n_runs,n_items))
auc = np.empty(shape=(n_runs,n_items))
loss = np.empty(shape=(n_runs,n_items))
precision = np.empty(shape=(n_runs,n_items))

for i in range(n_runs):
	with open(log_list[i]+'log_validation.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')  
		validation = np.array(list(reader)).astype(float)
		accuracy[i,:] = validation[0:n_items,0]	
		auc[i,:] = validation[0:n_items,1]
		loss[i,:] = validation[0:n_items,2]		
		precision[i,:] = validation[0:n_items,3]	

x = [i*interval for i  in range(n_items)]
# Plot Accuracy
plt.clf()   
plt.errorbar(x,np.mean(accuracy,axis=0),yerr=np.std(accuracy,axis=0),c='navy')
plt.title('Validation Accuracy')
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig(pdf_location+'validation_accuracy.pdf')
plt.savefig(png_location+'validation_accuracy.png')
# Plot AUC
plt.clf()   
plt.errorbar(x,np.mean(auc,axis=0),yerr=np.std(auc,axis=0),c='navy')
plt.title('Validation AUC')
plt.xlabel('Update')
plt.ylabel('AUC')
plt.tight_layout()
plt.savefig(pdf_location+'validation_auc.pdf')
plt.savefig(png_location+'validation_auc.png')
#Plot Precision
plt.clf()   
plt.errorbar(x,np.mean(precision,axis=0),yerr=np.std(precision,axis=0),c='navy')
plt.title('Validation Precision')
plt.xlabel('Update')
plt.ylabel('Precision')
plt.tight_layout()
plt.savefig(pdf_location+'validation_precision.pdf')
plt.savefig(png_location+'validation_precision.png')
# Plot Loss
plt.clf()   
plt.errorbar(x,np.mean(loss,axis=0),yerr=np.std(loss,axis=0),c='navy')
plt.title('Validation Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(pdf_location+'validation_loss.pdf')
plt.savefig(png_location+'validation_loss.png')


loss = np.empty(shape=(n_runs,(n_items-1)*interval))
for i in range(n_runs):
	with open(log_list[i]+'log_loss.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		training = np.array(list(reader)).astype(float)
		loss[i,:] = training[0:((n_items-1)*interval),0]	


x = [i for i  in range((n_items-1)*interval)]

plt.clf()   
plt.errorbar(x,np.mean(loss,axis=0),yerr=0,c='navy')
plt.title('Mean of Training Loss')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(pdf_location+'training_loss.pdf')
plt.savefig(png_location+'training_loss.png')

