import sys, os, time, datetime, csv, yaml, argparse
import numpy as np
from numpy import pi as PI
import tensorflow as tf
############################################################################################
def delete_all_logs(log_dir):
# Delete all .csv files in directory
	log_list = os.listdir(log_dir)
	for item in log_list:
		if item.endswith('.csv'):
			os.remove(log_dir+item)
			print(str(datetime.datetime.now()) + ' Deleted old log: ' + log_dir+item)
############################################################################################
def shape(tensor):
    s = tensor.get_shape()
    return s[i].value 
def log_tensor_array(tensor,log_dir,filename):
# Log 2D tensorflow array
	with open(log_dir + filename, 'a') as f:
		for i in range(tensor.shape[0]):
			for item in tensor[i].numpy():
					f.write('%.15f,' %item)
			f.write('\n')	
############################################################################################
def map2angle(arr0):
# Mapping the cylindrical coordinates to 0-2PI
	arr = np.zeros(arr0.shape)
	r_min     = 0.
	r_max     = 1.1
	phi_min   = -1.0
	phi_max   = 1.0
	z_min     = -1.1
	z_max     = 1.1
	arr[:,0] =  (arr0[:,0]-r_min)/(r_max-r_min) * 2 * PI
	arr[:,1] =  (arr0[:,1]-phi_min)/(phi_max-phi_min) * 2 * PI 
	arr[:,2] =  (arr0[:,2]-z_min)/(z_max-z_min) * 2 * PI
	mapping_check(arr)
	return arr
############################################################################################
def mapping_check(arr):
	for row in arr:
		for item in row:
			if (item > (2 * PI)) or (item < 0):
				raise ValueError('WARNING!: WRONG MAPPING!!!!!!')
############################################################################################
def preprocess(data):
	X,Ro,Ri,y  = data
	X 	       = tf.constant(X,dtype=tf.float64) # map2angle(X) with quantum circuits
	Ri         = tf.constant(Ri,dtype=tf.float64)
	Ro         = tf.constant(Ro,dtype=tf.float64)	
	edge_array = [X,Ri,Ro]
	return edge_array, tf.constant(y,dtype=tf.float64)
############################################################################################
def parse_args():
	parser = argparse.ArgumentParser(description='Load config file!')
	add_arg = parser.add_argument
	add_arg('config')
	return parser.parse_args()
############################################################################################
def load_config(args):
	with open(args.config, 'r') as ymlfile:
		config = yaml.load(ymlfile)
		print('Printing configs: ')
		for key in config:
			print(key + ': ' + str(config[key]))
		print('Log dir: ' + config['log_dir'])
		print('Training data input dir: ' + config['train_dir'])
		print('Validation data input dir: ' + config['train_dir'])
		delete_all_logs(config['log_dir'])
	# LOG the config (OPTIONAL)
	with open(config['log_dir'] + 'config.yaml', 'w') as f:
		for key in config:
			f.write('%s : %s \n' %(key,str(config[key])))
	return config
############################################################################################
def get_params(param_type):
	with open('params/test/'+param_type+'_params.csv', 'r') as f:
		reader = csv.reader(f, delimiter=',')
		return np.array(list(reader))[:,0:-1].astype(float)
############################################################################################
