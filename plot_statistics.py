import matplotlib.pyplot as plt 
import numpy as np
import csv

log_location = 'logs/log_lr_5/'
png_location = 'png/lr5/'
with open(log_location + 'log_loss.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	loss = np.array(list(reader)).astype(float)
with open(log_location+'log_validation.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	valid = np.array(list(reader)).astype(float) * 100
with open(log_location+'log_theta.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	theta = np.delete(np.array(list(reader)[:-2]),11,1).astype(float) 
# Plot
plt.clf()   
x = [(i+1) for i  in range(len(loss))]
plt.plot(x,loss,marker='o')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.savefig(png_location+'statistics_loss.png')

plt.clf()   
x = [i*50 for i  in range(len(valid))]
plt.plot(x,valid,marker='o')
plt.xlabel('Update')
plt.ylabel('Accuracy')
plt.savefig(png_location+'statistics_validation.png')

plt.clf()
x = [(i+1) for i in range(len(theta))]
for i in range(11):
	plt.plot(x,theta[:,i],marker='o',label=r'$\theta_{'+str(i)+'}$')
plt.xlabel('Update')
plt.ylabel(r'Angle (0 - 2$\pi$)')
plt.savefig(png_location+'statistics_angle.png')
