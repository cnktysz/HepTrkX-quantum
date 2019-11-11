import sys,os,time
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from datasets.hitgraphs import HitGraphDataset
import multiprocessing
from qnetworks.TTN import TTN_edge_forward, TTN_edge_back

def map2angle(B):
	# Maps input features to 0-2PI
	n_section = 8
	r_min 	  = 0
	r_max     = 1.
	phi_min   = -np.pi/n_section
	phi_max   = np.pi/n_section
	z_min 	  = -1.200
	z_max 	  = 1.200
	B[:,0] = 2*np.pi * (B[:,0]-r_min)/(r_max-r_min) 
	B[:,1] = 2*np.pi * (B[:,1]-phi_min)/(phi_max-phi_min) 
	B[:,2] = 2*np.pi * (B[:,2]-z_min)/(z_max-z_min) 
	B[:,3] = 2*np.pi * (B[:,3]-r_min)/(r_max-r_min) 
	B[:,4] = 2*np.pi * (B[:,4]-phi_min)/(phi_max-phi_min) 
	B[:,5] = 2*np.pi * (B[:,5]-z_min)/(z_max-z_min)
	return B
def gradient(edge_array,theta_learn,label,shots,noisy=False):
	out 	    = TTN_edge_forward(edge_array,theta_learn,shots=shots,noisy=noisy,properties=properties)
	error 		= out - label
	loss  	    = binary_cross_entropy(out,label)
	gradient 	= TTN_edge_back(edge_array,theta_learn,shots=shots,noisy=noisy,properties=properties)
	return error, loss, gradient
def binary_cross_entropy(output,label):
	return -(label*np.log(output+1e-6) + (1-label)*np.log(1-output+1e-6))
############################################################################################
##### MAIN ######
if __name__ == '__main__':
	# Random arrays
	theta_learn = np.array([0.59955824, 0.72293889, 0.76065828, 0.07674539, 0.22250796, 0.68152528, 0.72906506, 0.67450772, 0.51897852, 0.57968062, 0.3478734]) * 2 * np.pi
	edge_array = np.array([0.39437991, 0.58592331, 0.71819769, 0.07528009, 0.98749489, 0.21369549]) * 2 * np.pi 
	y = 0

	input_dir = 'data/hitgraphs_big'
	log_dir   = 'logs/gradient/noiseless/'
	png_dir   =	'png/gradient/noiseless/'
	n_files = 1

	provider = IBMQ.load_account()
	backends = provider.backends()
	device = provider.get_backend('ibmq_16_melbourne')
	properties = device.properties()
	
	noisy = False
	n_run  = 100
	shots  = [10, 100, 1000]
	range_ = np.zeros((11,2))
	
	for shot in shots:
		t0 = time.time()
		print('Testing shots = ' + str(shot))
		error_test    = np.zeros(n_run)
		loss_test     = np.zeros(n_run)
		gradient_test = np.zeros((n_run,11)) 
		for i in range(n_run):
			error_test[i],loss_test[i],gradient_test[i,:] = gradient(edge_array,theta_learn,y,shots=shot,noisy=noisy)
		with open(log_dir+'log_gradient_'+str(shot)+'_shots.csv', 'w') as f:
			for row in gradient_test:
				for item in row:
					f.write('%.4f,' % item)
				f.write('\n')
		with open(log_dir+'log_error_'+str(shot)+'_shots.csv', 'w') as f:
			for item in error_test:
				f.write('%.4f' % item)
			f.write('\n')
		with open(log_dir+'log_loss_'+str(shot)+'_shots.csv', 'w') as f:
			for item in loss_test:
				f.write('%.4f' % item)
			f.write('\n')		
			
		# Plot the results	
		plt.clf()	
		_ = plt.hist(error_test,bins='auto')
		plt.xlabel('Error')
		plt.title('$\mu= $'+ str(round(error_test.mean(),3)) + ', std= ' + str(round(error_test.std(),3)))
		plt.tight_layout()
		plt.savefig(png_dir+'test_error_'+str(shot)+'shots_.pdf')

		plt.clf()	
		_ = plt.hist(loss_test,bins='auto')
		plt.title('$\mu= $'+ str(round(loss_test.mean(),3)) + ', std= ' + str(round(loss_test.std(),3)))
		plt.xlabel('Loss')
		plt.tight_layout()
		plt.savefig(png_dir+'test_loss_'+str(shot)+'shots_.pdf')

		for i in range(11):
			if shot == min(shots):
				range_[i,:] = [min(gradient_test[:,i]),max(gradient_test[:,i])]
			plt.clf()	
			_ = plt.hist(gradient_test[:,i],bins=20,range=range_[i,:])
			plt.title('Gradient of '+str(i)+'th angle: ' + r'$\mu= $'+ str(round(gradient_test[:,i].mean(),3)) + r'$, \sigma= $' + str(round(gradient_test[:,i].std(),3)))
			plt.tight_layout()
			plt.savefig(png_dir+'test_gradient_'+str(i)+'_'+str(shot)+'shots_.pdf')
		
		# Print Summary	
		duration = time.time() - t0
		print('Mean error: %.3f, Mean Loss: %.3f, Elapsed: %dm%ds ' % (error_test.mean(),loss_test.mean(),duration/60,duration%60))

