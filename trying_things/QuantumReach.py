import numpy as np
import matplotlib.pyplot as plt
import sys,os,time
sys.path.append(os.path.abspath(os.path.join('.')))
from qnetworks.TTN import TTN_edge_forward



def add2dic(dic1,dic2):
	for keys in dic1:
		if keys not in dic2.keys():
			dic2[keys] = dic1[keys]
		else:
			dic2[keys] += dic1[keys]
	return dic2

def scan_hilbert_space(n_span=1,phase_shift=0):
	results = []
	theta_learn = np.zeros(11)
	for t0 in range(n_span):
		for t1 in range(n_span):
			for t2 in range(n_span):
				for t3 in range(n_span):
					for t4 in range(n_span):
						for t5 in range(n_span):
							for t6 in range(n_span):
								for t7 in range(n_span):
									for t8 in range(n_span):
										for t9 in range(n_span):
											for t10 in range(n_span):
												theta_learn[0] = t0 * 2*np.pi / n_span + phase_shift
												theta_learn[1] = t1 * 2*np.pi / n_span + phase_shift
												theta_learn[2] = t2 * 2*np.pi / n_span + phase_shift
												theta_learn[3] = t3 * 2*np.pi / n_span + phase_shift
												theta_learn[4] = t4 * 2*np.pi / n_span + phase_shift
												theta_learn[5] = t5 * 2*np.pi / n_span + phase_shift
												theta_learn[6] = t6 * 2*np.pi / n_span + phase_shift
												theta_learn[7] = t7 * 2*np.pi / n_span + phase_shift 
												theta_learn[8] = t8 * 2*np.pi / n_span + phase_shift
												theta_learn[9] = t9 * 2*np.pi / n_span + phase_shift
												theta_learn[10] = t10 * 2*np.pi / n_span + phase_shift
												out = TTN_edge_forward(np.zeros(6),theta_learn,shots=1000,noisy=True)
												results.append(out)

	return results


results = scan_hilbert_space(n_span=2,phase_shift=np.pi/8)

#plt.bar(total_dict.keys(),total_dict.values())
plt.hist(results,bins=[i/20 for i in range(20)])
plt.savefig('png/QuantumReach.png')
plt.show()







