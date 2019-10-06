import matplotlib.pyplot as plt 
import numpy as np

# Plot the result every update  
plt.clf()   
x = [(i+1) for i  in range(n_file+1)]
plt.plot(x,loss_log[:n_file+1],marker='o')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.savefig('png\statistics_loss.png')

plt.clf()
for i in range(11):
	plt.plot(x,theta_log[:n_file+1,i],marker='o',label=r'$\theta_{'+str(i)+'}$')
plt.xlabel('Update')
plt.ylabel(r'Angle (0 - 2$\pi$)')
plt.savefig('png\statistics_angle.png')