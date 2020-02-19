import os,csv,datetime
import numpy as np
from numpy import pi as PI
def delete_all_logs(log_dir):
	log_list = os.listdir(log_dir)
	for item in log_list:
		if item.endswith('.csv'):
			os.remove(log_dir+item)
			print(str(datetime.datetime.now()) + ' Deleted old log: ' + log_dir+item)
def log_tensor_array(tensor,log_dir,filename):
	with open(log_dir + filename, 'a') as f:
		for i in range(len(tensor)):
			for item in tensor[i].numpy():
				f.write('%.4f,' %item)
		f.write('\n')	
def map2angle(arr0):
	# Maps input features to 0-2PI
	# This might depend on the data. BE CAREFUL!
	arr = np.zeros(arr0.shape)
	r_min     = 0.
	r_max     = 1.1
	phi_min   = -1.
	phi_max   = 1.
	z_min     = -1.1
	z_max     = 1.1
	arr[:,0] =  (arr0[:,0]-r_min)/(r_max-r_min) * 2 * PI
	arr[:,1] =  (arr0[:,1]-phi_min)/(phi_max-phi_min) * 2 * PI 
	arr[:,2] =  (arr0[:,2]-z_min)/(z_max-z_min) * 2 * PI
	mapping_check(arr)
	return arr
def mapping_check(arr):
	for row in arr:
		for item in row:
			if (item > (2 * PI)) or (item < 0):
				print('WARNING!: WRONG MAPPING!!!!!!')
				