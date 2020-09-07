import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import csv
from sklearn import metrics
import sys


log_loc1 = 'logs/TTN/TTN_hid1_it1/'
log_loc2 = 'logs/MERA/MERA_hid1_it1/'
log_loc0 = 'logs/MPS/MPS_hid1_it1/'

label1 = r'$TTN$'
label2 = r'$MERA$'
label0 = r'$MPS$'

pdf_location = 'pdf/compare_multiple/new/'
png_location = 'png/compare_multiple/new/'


'''
log_loc0 = 'logs/TTN/old_mapping_0_2pi/TTN_hid1_it1/'
log_loc1 = 'logs/TTN/old_mapping_0_2pi/TTN_hid1_it2/'
log_loc2 = 'logs/TTN/old_mapping_0_2pi/TTN_hid2_it2/'

label0 = r'$N_{Dim}$' + ' = 1, ' +r'$N_{it}$' + ' = 1'
label1 = r'$N_{Dim}$' + ' = 1, ' +r'$N_{it}$' + ' = 2'
label2 = r'$N_{Dim}$' + ' = 2, ' +r'$N_{it}$' + ' = 2'

pdf_location = 'pdf/compare_multiple/TTN/old/'
png_location = 'png/compare_multiple/TTN/old/'
'''

n_runs0 = 3
n_runs1 = 3
n_runs2 = 3

log_list0 = [log_loc0 + 'run' + str(i+1) + '/' for i in range(n_runs0)]
log_list1 = [log_loc1 + 'run' + str(i+1) + '/' for i in range(n_runs1)]
log_list2 = [log_loc2 + 'run' + str(i+1) + '/' for i in range(n_runs2)]


print('Plots will be saved to:')
print('PDF: ' + pdf_location)
print('PNG: ' + png_location)

interval = 50

def file_length(fname):
        with open(fname) as f:
                for i, l in enumerate(f):
                        pass
        return i + 1


n_items0 = file_length(log_list0[0]+'log_validation.csv')
n_items1 = file_length(log_list1[0]+'log_validation.csv')
n_items2 = file_length(log_list2[0]+'log_validation.csv')
# length of the arrays
accuracy0 = np.empty(shape=(n_runs0,n_items0))
auc0 = np.empty(shape=(n_runs0,n_items0))
loss0 = np.empty(shape=(n_runs0,n_items0))
precision0 = np.empty(shape=(n_runs0,n_items0))

accuracy1 = np.empty(shape=(n_runs1,n_items1))
auc1 = np.empty(shape=(n_runs1,n_items1))
loss1 = np.empty(shape=(n_runs1,n_items1))
precision1 = np.empty(shape=(n_runs1,n_items1))

accuracy2 = np.empty(shape=(n_runs2,n_items2))
auc2 = np.empty(shape=(n_runs2,n_items2))
loss2 = np.empty(shape=(n_runs2,n_items2))
precision2 = np.empty(shape=(n_runs2,n_items2))

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

for i in range(n_runs2):
	with open(log_list2[i]+'log_validation.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')  
		validation = np.array(list(reader)).astype(float)
		accuracy2[i,:] = validation[0:n_items2,0]	
		auc2[i,:] = validation[0:n_items2,1]
		loss2[i,:] = validation[0:n_items2,2]		
		precision2[i,:] = validation[0:n_items2,3]	

x0 = [i*interval for i  in range(n_items0)]
x1 = [i*interval for i  in range(n_items1)]
x2 = [i*interval for i  in range(n_items2)]

font = {
		'size'   : 16,
	}

axes = {
	    'titlesize' : 16,
		'labelsize' : 16,
	}

matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)

# Plot Accuracy
plt.clf()   
plt.errorbar(x0,np.mean(accuracy0,axis=0),yerr=np.std(accuracy0,axis=0),c='navy', label=label0)
plt.errorbar(x1,np.mean(accuracy1,axis=0),yerr=np.std(accuracy1,axis=0),c='darkorange', label=label1)
plt.errorbar(x2,np.mean(accuracy2,axis=0),yerr=np.std(accuracy2,axis=0),c='red', label=label2)
plt.title('Validation Accuracy')
plt.xlabel('Update (1 epoch = 1400)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(pdf_location+'validation_accuracy.pdf')
plt.savefig(png_location+'validation_accuracy.png')
# Plot AUC
plt.clf()   
plt.errorbar(x0,np.mean(auc0,axis=0),yerr=np.std(auc0,axis=0),c='navy', label=label0)
plt.errorbar(x1,np.mean(auc1,axis=0),yerr=np.std(auc1,axis=0),c='darkorange', label=label1)
plt.errorbar(x2,np.mean(auc2,axis=0),yerr=np.std(auc2,axis=0),c='red', label=label2)
plt.title('Validation AUC')
plt.xlabel('Update (1 epoch = 1400)')
plt.ylabel('AUC')
#plt.ylim(0.4,0.8)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(pdf_location+'validation_auc.pdf')
plt.savefig(png_location+'validation_auc.png')
#Plot Precision
plt.clf()   
plt.errorbar(x0,np.mean(precision0,axis=0),yerr=np.std(precision0,axis=0),c='navy', label=label0)
plt.errorbar(x1,np.mean(precision1,axis=0),yerr=np.std(precision1,axis=0),c='darkorange', label=label1)
plt.errorbar(x2,np.mean(precision2,axis=0),yerr=np.std(precision2,axis=0),c='red', label=label2)
plt.title('Validation Precision')
plt.xlabel('Update (1 epoch = 1400)')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(pdf_location+'validation_precision.pdf')
plt.savefig(png_location+'validation_precision.png')
# Plot Loss
plt.clf()   
plt.errorbar(x0,np.mean(loss0,axis=0),yerr=np.std(loss0,axis=0),c='navy',label=label0)
plt.errorbar(x1,np.mean(loss1,axis=0),yerr=np.std(loss1,axis=0),c='darkorange',label=label1)
plt.errorbar(x2,np.mean(loss2,axis=0),yerr=np.std(loss2,axis=0),c='red',label=label2)
plt.title('Validation Loss')
plt.xlabel('Update (1 epoch = 1400)')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(pdf_location+'validation_loss.pdf')
plt.savefig(png_location+'validation_loss.png')


loss0 = np.empty(shape=(n_runs0,(n_items0-1)*interval))
loss1 = np.empty(shape=(n_runs1,(n_items1-1)*interval))
loss2 = np.empty(shape=(n_runs2,(n_items2-1)*interval))
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
for i in range(n_runs2):
	with open(log_list2[i]+'log_loss.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		training = np.array(list(reader)).astype(float)
		loss2[i,:] = training[0:((n_items2-1)*interval),0]	

x0 = [i for i  in range((n_items0-1)*interval)]
x1 = [i for i  in range((n_items1-1)*interval)]
x2 = [i for i  in range((n_items2-1)*interval)]

plt.clf()   
plt.errorbar(x0,np.mean(loss0,axis=0),yerr=0,c='navy',label=label0)
plt.errorbar(x1,np.mean(loss1,axis=0),yerr=0,c='darkorange', label=label1)
plt.errorbar(x2,np.mean(loss2,axis=0),yerr=0,c='red', label=label2)
plt.title('Mean of Training Loss')
plt.xlabel('Update (1 epoch = 1400)')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(pdf_location+'training_loss.pdf')
plt.savefig(png_location+'training_loss.png')

