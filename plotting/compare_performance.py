import matplotlib.pyplot as plt 
import numpy as np
import csv
from sklearn import metrics
import sys

log_loc0 = 'logs/TTN/TTN_hid1_it1/'
log_loc1 = 'logs/MERA/MERA_hid1_it1/'
log_loc5 = 'logs/MPS/MPS_hid1_it1/'

log_loc2 = 'logs/CGNN/CGNN_hid1_it1/'
log_loc3 = 'logs/CGNN/CGNN_hid5_it1/'
log_loc4 = 'logs/CGNN/CGNN_hid10_it1/'

log_list0 =[
	log_loc5,
	log_loc0,
	log_loc1,
	]

log_list1 =[
	log_loc2,
	log_loc3,
	log_loc4,
	]


pdf_location = 'pdf/'
png_location = 'png/'


print('Plots will be saved to:')
print('PDF: ' + pdf_location)
print('PNG: ' + png_location)

def read_multiple_files(filename, n_runs, n_items):
	accuracy  = np.empty(shape=(n_runs,n_items))
	auc       = np.empty(shape=(n_runs,n_items))
	loss      = np.empty(shape=(n_runs,n_items))
	precision = np.empty(shape=(n_runs,n_items))
	log_list = [filename + 'run' + str(i+1) + '/' for i in range(n_runs)]
	for i in range(n_runs):
		with open(log_list[i]+'log_validation.csv', 'r') as f:
			reader = csv.reader(f, delimiter=',')  
			validation = np.array(list(reader)).astype(float)
			accuracy[i,:] = validation[0:n_items,0]	
			auc[i,:] = validation[0:n_items,1]
			loss[i,:] = validation[0:n_items,2]		
			precision[i,:] = validation[0:n_items,3]
	return accuracy, auc, loss, precision


# read last items
q_auc_std  =  []
q_auc_mean = []
q_loss_std  =  []
q_loss_mean = []
for log_loc in log_list0:
	if log_loc == log_loc0: n_items=5
	if log_loc == log_loc1: n_items=3
	if log_loc == log_loc5: n_items=2
	_, auc, loss, _ = read_multiple_files(log_loc, n_runs=3, n_items=n_items)
	q_auc_mean = np.append(q_auc_mean, np.mean(auc[:,-1],axis=0))
	q_auc_std  = np.append(q_auc_std, np.std(auc[:,-1],axis=0))
	q_loss_mean = np.append(q_loss_mean, np.mean(loss[:,-1],axis=0))
	q_loss_std  = np.append(q_loss_std, np.std(loss[:,-1],axis=0))

c_auc_std  =  []
c_auc_mean = []
c_loss_std  =  []
c_loss_mean = []
for log_loc in log_list1:
	_, auc, loss, _ = read_multiple_files(log_loc, n_runs=3, n_items=29)
	c_auc_mean = np.append(c_auc_mean, np.mean(auc[:,-1],axis=0))
	c_auc_std  = np.append(c_auc_std, np.std(auc[:,-1],axis=0))
	c_loss_mean = np.append(c_loss_mean, np.mean(loss[:,-1],axis=0))
	c_loss_std  = np.append(c_loss_std, np.std(loss[:,-1],axis=0))

c_n_params = [30, 266, 831]
q_n_params = [40, 42, 58] 

# Plot AUC
plt.clf()   
plt.errorbar(c_n_params,c_auc_mean,yerr=c_auc_std,marker="o", c='navy', label='classical')
plt.errorbar(q_n_params,q_auc_mean,yerr=q_auc_std,marker="o",linestyle="None", c='darkorange', label='quantum')
plt.text(q_n_params[0],q_auc_mean[0],s='MPS-hid1')
plt.text(q_n_params[1],q_auc_mean[1],s='TTN-hid1')
plt.text(q_n_params[2],q_auc_mean[2],s='MERA-hid1')
plt.text(c_n_params[0],c_auc_mean[0]+0.01,s='HepTrkX-hid1')
plt.text(c_n_params[1]-100,c_auc_mean[1]-0.015,s='HepTrkX-hid5')
plt.text(c_n_params[2]-400,c_auc_mean[2]-0.01,s='HepTrkX-hid10')

plt.title('AUC Comparison after 1 epoch')
plt.xlabel('# Parameters')
plt.ylabel('AUC')
plt.xscale('log')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.savefig(pdf_location+'validation_comparison.pdf')
plt.savefig(png_location+'validation_comparison.png')





