# Author: Cenk Tüysüz
# Date: 16.03.2020
# Script to divide data into 2 folders

import os
from random import shuffle
from shutil import copyfile

in_loc  = '/home/ctuysuz/Repos/HepTrkX-quantum/data/hitgraphs_big/'
out_loc = '/home/ctuysuz/Repos/HepTrkX-quantum/data/graph_data/'

print('Reading data from: ' + in_loc)

dirs = os.listdir(out_loc)
if 'train' in dirs:
	t = os.listdir(out_loc+'train/')
	for file in t:
		os.remove(out_loc+'train/'+file)
else:
	os.mkdir(out_loc+'train')
if 'valid' in dirs:
	v = os.listdir(out_loc+'valid/')
	for file in v:
		os.remove(out_loc+'valid/'+file)
else:
	os.mkdir(out_loc+'valid')

ratio   = 9 
files = os.listdir(in_loc)
shuffle(files)
n_files = len(files)

n_valid = 200 # CHANGE n_valid FOR CUSTOM SPLITTING 
n_train = n_files - n_valid   

f_train = files[:n_train]
f_valid = files[n_train:]

for file in f_train:
	copyfile(in_loc+file,out_loc+'train/'+file)
for file in f_valid:
	copyfile(in_loc+file,out_loc+'valid/'+file)

t_files = os.listdir(out_loc+'train')
v_files = os.listdir(out_loc+'valid')
print(str(len(t_files)) + ' files copied to ' + out_loc+'train/')
print(str(len(v_files)) + ' files copied to ' + out_loc+'valid/')
print('Divided the dataset of ' + str(n_files) + ' with ratio ' + str(len(t_files)/len(v_files)) +' to 1')