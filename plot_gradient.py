import matplotlib.pyplot as plt 
import numpy as np
import csv

log_location = 'logs/gradient/'
png_location = 'png/gradient/'
with open(log_location + 'log_gradient_10_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	grad_10 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
with open(log_location + 'log_gradient_100_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	grad_100 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
with open(log_location + 'log_gradient_1000_shots.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	grad_1000 = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 	
	

range_ = [min(grad_10[:,0]),max(grad_10[:,0])]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,sharey=True,figsize=(10,3))
_ = ax1.hist(grad_10[:,0],bins=20,range=range_)
ax1.set(title=r'$\mu= $'+ str(round(grad_10[:,0].mean(),3)) + r'$, \sigma= $' + str(round(grad_10[:,0].std(),3)),xlabel='Gradient',ylabel='Counts' )
_ = ax2.hist(grad_100[:,0],bins=20,range=range_)
ax2.set(title=r'$\mu= $'+ str(round(grad_100[:,0].mean(),3)) + r'$, \sigma= $' + str(round(grad_100[:,0].std(),3)),xlabel='Gradient',ylabel='Counts' )
_ = ax3.hist(grad_1000[:,0],bins=20,range=range_)
ax3.set(title=r'$\mu= $'+ str(round(grad_1000[:,0].mean(),3)) + r'$, \sigma= $' + str(round(grad_1000[:,0].std(),3)),xlabel='Gradient',ylabel='Counts' )
plt.tight_layout()


plt.savefig(png_location+'gradients.pdf')

"""
range_ = [min(grad_10[:,0]),max(grad_10[:,0])]
plt.clf()	
_ = plt.hist(grad_10[:,0],bins=20,range=range_)
plt.title('Gradient of 0th angle: ' + r'$\mu= $'+ str(round(grad_10[:,0].mean(),3)) + r'$, \sigma= $' + str(round(grad_10[:,0].std(),3)))
plt.xlabel('Gradient')
plt.ylabel('Counts')
plt.savefig(png_location+'gradient_'+str(10)+'_shots.pdf')
	
plt.clf()	
_ = plt.hist(grad_100[:,0],bins=20,range=range_)
plt.title('Gradient of 0th angle: ' + r'$\mu= $'+ str(round(grad_100[:,0].mean(),3)) + r'$, \sigma= $' + str(round(grad_100[:,0].std(),3)))
plt.xlabel('Gradient')
plt.ylabel('Counts')
plt.savefig(png_location+'gradient_'+str(100)+'_shots.pdf')

plt.clf()	
_ = plt.hist(grad_1000[:,0],bins=20,range=range_)
plt.title('Gradient of 0th angle: ' + r'$\mu= $'+ str(round(grad_1000[:,0].mean(),3)) + r'$, \sigma= $' + str(round(grad_1000[:,0].std(),3)))
plt.xlabel('Gradient')
plt.ylabel('Counts')
plt.savefig(png_location+'gradient_'+str(1000)+'_shots.pdf')		
"""