import matplotlib.pyplot as plt 
import numpy as np
import csv

noisy_log_location = 'logs/gradient/noisy/'
noiseless_log_location = 'logs/gradient/noiseless/'
png_location = 'png/gradient/'
with open(noisy_log_location + 'log_gradient_10_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	noisy_grad_10 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
with open(noisy_log_location + 'log_gradient_100_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	noisy_grad_100 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
with open(noisy_log_location + 'log_gradient_1000_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	noisy_grad_1000 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
with open(noiseless_log_location + 'log_gradient_10_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	noiseless_grad_10 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
with open(noiseless_log_location + 'log_gradient_100_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	noiseless_grad_100 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
with open(noiseless_log_location + 'log_gradient_1000_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	noiseless_grad_1000 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float)
	
for i in range(11):	
	range_ = [min(np.minimum(noiseless_grad_10[:,i],noisy_grad_10[:,i])),max(np.maximum(noiseless_grad_10[:,i],noisy_grad_10[:,i]))]
	fig, axes = plt.subplots(2, 3,sharey=True,figsize=(10,3))
	_ = axes[0,0].hist(noiseless_grad_10[:,i],bins=20,range=range_,label='Noise: OFF \nShots: 10')
	axes[0,0].set(title=r'$\mu= $'+ str(round(noiseless_grad_10[:,i].mean(),3)) + r'$, \sigma= $' + str(round(noiseless_grad_10[:,i].std(),3)),xlabel=r'$\partial \^y / \partial \theta_{{{}}} $'.format(i), ylabel='Counts' )
	_ = axes[0,1].hist(noiseless_grad_100[:,i],bins=20,range=range_,label='Noise: OFF \nShots: 100')
	axes[0,1].set(title=r'$\mu= $'+ str(round(noiseless_grad_100[:,i].mean(),3)) + r'$, \sigma= $' + str(round(noiseless_grad_100[:,i].std(),3)),xlabel=r'$\partial \^y / \partial \theta_{{{}}} $'.format(i), ylabel='Counts' )
	_ = axes[0,2].hist(noiseless_grad_1000[:,i],bins=20,range=range_,label='Noise: OFF \nShots: 1000')
	axes[0,2].set(title=r'$\mu= $'+ str(round(noiseless_grad_1000[:,i].mean(),3)) + r'$, \sigma= $' + str(round(noiseless_grad_1000[:,i].std(),3)),xlabel=r'$\partial \^y / \partial \theta_{{{}}} $'.format(i), ylabel='Counts' )
	_ = axes[1,0].hist(noisy_grad_10[:,i],bins=20,range=range_,label='Noise: ON \nShots: 10')
	axes[1,0].set(title=r'$\mu= $'+ str(round(noisy_grad_10[:,i].mean(),3)) + r'$, \sigma= $' + str(round(noisy_grad_10[:,i].std(),3)),xlabel=r'$\partial \^y / \partial \theta_{{{}}} $'.format(i), ylabel='Counts' )
	_ = axes[1,1].hist(noisy_grad_100[:,i],bins=20,range=range_,label='Noise: ON \nShots: 100')
	axes[1,1].set(title=r'$\mu= $'+ str(round(noisy_grad_100[:,i].mean(),3)) + r'$, \sigma= $' + str(round(noisy_grad_100[:,i].std(),3)),xlabel=r'$\partial \^y / \partial \theta_{{{}}} $'.format(i), ylabel='Counts' )
	_ = axes[1,2].hist(noisy_grad_1000[:,i],bins=20,range=range_,label='Noise: ON \nShots: 1000')
	axes[1,2].set(title=r'$\mu= $'+ str(round(noisy_grad_1000[:,i].mean(),3)) + r'$, \sigma= $' + str(round(noisy_grad_1000[:,i].std(),3)),xlabel=r'$\partial \^y / \partial \theta_{{{}}} $'.format(i), ylabel='Counts' )
	
	#legend settings
	for k in range(2):
		for l in range(3):
			leg = axes[k,l].legend(bbox_to_anchor=(-0.14, 1.1),loc='upper left',frameon=False,prop={'size': 8})
			for text in leg.get_texts():
				text.set_color('r')
				text.set_fontweight('bold')
				text.set_horizontalalignment('left')
			for item in leg.legendHandles:
				item.set_visible(False)

	plt.tight_layout()
	plt.savefig(png_location+'gradients_of_' + str(i) + 'th_angle.pdf')
