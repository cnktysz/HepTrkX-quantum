import sys, os, time, datetime, csv, yaml, argparse
import numpy as np
from numpy import pi as PI
from collections import namedtuple

######################## FUNCTIONS & CLASSES FOR READING GRAPH DATA ########################

# This part of the tools is inherited mostly from Hep.TrkX: https://github.com/HEPTrkX/heptrkx-gnn-tracking

# A Graph is a namedtuple of matrices (X, Ri, Ro, y)
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])

class GraphDataset():
    def __init__(self, input_dir, n_samples=None):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.startswith('event') and f.endswith('.npz')]
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)

def get_dataset(input_dir,n_files):
    return GraphDataset(input_dir, n_files)
def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))
def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.uint8):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y)

######################################## OTHER TOOLS #######################################
def delete_all_logs(log_dir):
# Delete all .csv files in directory
	log_list = os.listdir(log_dir)
	for item in log_list:
		if item.endswith('.csv'):
			os.remove(log_dir+item)
			print(str(datetime.datetime.now()) + ' Deleted old log: ' + log_dir+item)
############################################################################################
def log_tensor_array(tensor,log_dir,filename):
	# Log 2D tensorflow array
	with open(log_dir + filename, 'a') as f:
		for i in range(tensor.shape[0]):
			for item in tensor[i].numpy():
				f.write('%f,' %item)
			f.write('\n')
############################################################################################
def map2angle(arr0):
# Mapping the cylindrical coordinates to 0-4PI
	arr = np.zeros(arr0.shape)
	r_min     = 0.
	r_max     = 1.1
	phi_min   = -1.0
	phi_max   = 1.0
	z_min     = 0
	z_max     = 1.1
	arr[:,0] =  PI * (arr0[:,0]-r_min)/(r_max-r_min)          
	arr[:,1] =  PI * (arr0[:,1]-phi_min)/(phi_max-phi_min) 
	arr[:,2] =  PI * (np.abs(arr0[:,2])-z_min)/(z_max-z_min)  # take abs of z due to symmetry of z
	mapping_check(arr)
	return arr
############################################################################################
def mapping_check(arr):
# check if every element of the input array is within limits [0,2*pi]
	for row in arr:
		for item in row:
			if (item > (PI)) or (item < 0):
				raise ValueError('WARNING!: WRONG MAPPING!!!!!!')
############################################################################################
def preprocess(data):
	import tensorflow as tf
	X,Ro,Ri,y  = data 		      						    # decompose the event graph
	X 	       = tf.constant(map2angle(X),dtype=tf.float64) # map all coordinates to [0,2*pi]
	Ri         = tf.constant(Ri,dtype=tf.float64)           # Ri is converted to tf.constant 
	Ro         = tf.constant(Ro,dtype=tf.float64)	        # Ro is converted to tf. constant
	graph_array = [X,Ri,Ro]                                 # construct the event graph again
	return graph_array, tf.constant(y, dtype=tf.float64)    # return event graph and labels
############################################################################################
def parse_args():
	# generic parser, nothing fancy here
	parser = argparse.ArgumentParser(description='Load config file!')
	add_arg = parser.add_argument
	add_arg('config')
	return parser.parse_args()
############################################################################################
def load_config(args):
	# read the config file 
	with open(args.config, 'r') as ymlfile:
		config = yaml.load(ymlfile)
		# print all configs
		print('Printing configs: ')
		for key in config:
			print(key + ': ' + str(config[key]))
		print('Log dir: ' + config['log_dir'])
		print('Training data input dir: ' + config['train_dir'])
		print('Validation data input dir: ' + config['train_dir'])
		delete_all_logs(config['log_dir'])
	# LOG the config every time
	with open(config['log_dir'] + 'config.yaml', 'w') as f:
		for key in config:
			f.write('%s : %s \n' %(key,str(config[key])))
	# return the config dictionary
	return config
############################################################################################
def get_params(param_type,config):
	# read parameters of networks from a file specified below
	# parameters are created using tools/init_params.py
	if config['run_type'] == 'new_run':  # load params from params directory to initialize a network
		with open(config['param_dir'] + 'QGNN' + str(config['hid_dim']) + '/params_' + param_type + '.csv', 'r') as f:
			reader = csv.reader(f, delimiter=',')
			return np.array(list(reader))[:,0:-1].astype(float)
	# NOT TESTED YET!
	elif config['run_type'] == 'recovery_run': # load params to continue an aborted job to initialize a network
		with open(config['log_dir']+param_type+'log_params_' + param_type + '.csv', 'r') as f:
			reader = csv.reader(f, delimiter=',')
			return np.array(list(reader))[-1:,0:-1].astype(float)

	else: 
		RaiseValueError('Wrong paramater setting chosen or not implemented yet!')
############################################################################################












