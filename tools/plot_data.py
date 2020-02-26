import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
import matplotlib.pyplot as plt
from datasets.graph import load_graph
from tools import *
from tqdm import tqdm


if __name__ == '__main__':

    input_dir = 'data/hitgraphs_big'
    pdf_dir = 'pdf/data/'
    n_section = 8
    n_files = 16*100
   
    input_dir = os.path.expandvars(input_dir)
    filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if f.startswith('event') and f.endswith('.npz')])
    filenames[:n_files] if n_files is not None else filenames

    n_bins = 20
    r_combined = []
    p_combined = []
    z_combined = []
    y_combined = []
    for n_file in tqdm(range(n_files)):
        X, Ri, Ro, y = load_graph(filenames[n_file])
        r_combined = np.append(r_combined,X[:,0])
        p_combined = np.append(p_combined,X[:,1]*np.pi/n_section)
        z_combined = np.append(z_combined,X[:,2])
        y_combined = np.append(y_combined,y)

    print('Min r: %.2f, Max r: %.2f' %(np.min(r_combined), np.max(r_combined)))
    print('Min phi: %.2f, Max phi: %.2f' %(np.min(p_combined), np.max(p_combined)))
    print('Min z: %.2f, Max z: %.2f' %(np.min(z_combined), np.max(z_combined)))

    fig, axs = plt.subplots(1, 3, figsize = (10,4),sharey=True, tight_layout=True)
		
    axs[0].hist(r_combined, bins=[0.03,0.04,0.07,0.08,0.11,0.12,0.17,0.18,0.255,0.265,0.355,0.365,0.495,0.505,0.655,0.665,0.815,0.825,1.015,1.025])
    axs[1].hist(p_combined, bins=n_bins)
    axs[2].hist(z_combined, bins=n_bins)

    axs[0].set_xlabel('$r[m] $')
    axs[1].set_xlabel('$\Phi $')
    axs[2].set_xlabel('$z[m] $')
    plt.savefig(pdf_dir+'data_preprocessed.pdf')

    r_combined = []
    p_combined = []
    z_combined = []
    y_combined = []

    for n_file in tqdm(range(n_files)):
        X, Ri, Ro, y = load_graph(filenames[n_file])
        X_mapped = map2angle(X)
        r_combined = np.append(r_combined,X_mapped[:,0])
        p_combined = np.append(p_combined,X_mapped[:,1])
        z_combined = np.append(z_combined,X_mapped[:,2])
        y_combined = np.append(y_combined,y)

    print('Min r: %.2f, Max r: %.2f' %(np.min(r_combined), np.max(r_combined)))
    print('Min phi: %.2f, Max phi: %.2f' %(np.min(p_combined), np.max(p_combined)))
    print('Min z: %.2f, Max z: %.2f' %(np.min(z_combined), np.max(z_combined)))

    fig, axs = plt.subplots(1, 3, figsize = (10,4),sharey=True, tight_layout=True)
		
    axs[0].hist(r_combined, bins=2*np.pi*np.array([0.03,0.04,0.07,0.08,0.11,0.12,0.17,0.18,0.255,0.265,0.355,0.365,0.495,0.505,0.655,0.665,0.815,0.825,1.015,1.025]))
    axs[1].hist(p_combined, bins=n_bins)
    axs[2].hist(z_combined, bins=n_bins)

    axs[0].set_xlabel('$r[m] $')
    axs[1].set_xlabel('$\Phi $')
    axs[2].set_xlabel('$z[m] $')
    plt.savefig(pdf_dir+'data_mapped.pdf')




