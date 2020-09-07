import csv
import numpy as np 


file_list = ['logs/MPS/MPS_hid1_it1/run1/summary.csv', 'logs/MPS/MPS_hid1_it1/run2/summary.csv', 'logs/MPS/MPS_hid1_it1/run3/summary.csv'] 
#file_list = ['logs/TTN/TTN_hid1_it1/run1/summary.csv', 'logs/TTN/TTN_hid1_it1/run2/summary.csv', 'logs/TTN/TTN_hid1_it1/run3/summary.csv'] 
#file_list = ['logs/MERA/MERA_hid1_it1/run1/summary.csv', 'logs/MERA/MERA_hid1_it1/run2/summary.csv', 'logs/MERA/MERA_hid1_it1/run3/summary.csv'] 


def get_minute(str_):
	a = ''
	for s in str_:
		if s=='m': break
		a+=s
	return int(a)*60

def get_second(str_):
	a = ''
	flag = False
	for s in str_:
		if flag: a+=s
		if s == 'm': flag = True
	return int(a)	

for idx, file in enumerate(file_list):
	with open(file, 'r') as f:
		reader = csv.reader(f, delimiter=',')  
		a = np.array(list(reader))[:,-1]
		n_items = len(a)
	break

print(n_items)
times = np.zeros((len(file_list),n_items))
for idx, file in enumerate(file_list):
	with open(file, 'r') as f:
		reader = csv.reader(f, delimiter=',')  
		a = np.array(list(reader))[:,-1]
	b = np.array([item[10:-1] for item in a])

	for idy, item in enumerate(b):
		times[idx,idy]= get_minute(item)+get_second(item)

print(times.flatten())
mean = np.mean(times.flatten())
std   = np.std(times.flatten())
print('Mean time in seconds: %.2f, Std: %.2f'%(mean, std))
print('Mean time in minutes: %.2f, Std: %.2f'%(mean/60, std/60))


