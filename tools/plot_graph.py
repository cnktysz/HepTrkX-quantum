import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from datasets.hitgraphs import HitGraphDataset
from datasets.graph import load_graph
from tqdm import tqdm

def plot_cylindrical(filenames,n_section):

        print('Plotting file: ' + filenames[1] + ' to: ' + pdf_dir)
        X, Ri, Ro, y = load_graph(filenames[1])

        feats_o = X[np.where(Ri.T)[1]]
        feats_i = X[np.where(Ro.T)[1]]
        
        feats_o[:,0] = feats_o[:,0]*1000 
        feats_o[:,1] = feats_o[:,1]*np.pi/n_section
        feats_o[:,2] = feats_o[:,2]*1000  
        feats_i[:,0] = feats_i[:,0]*1000  
        feats_i[:,1] = feats_i[:,1]*np.pi/n_section
        feats_i[:,2] = feats_i[:,2]*1000  
        
        print('Plotting: After Preprocessing in Cylindrical coordinates!')
        
        fig, ax = plt.subplots(1, 2, figsize = (10,5),sharey=True, tight_layout=True)

        cmap = plt.get_cmap('bwr_r')
        ax[0].scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=1.)
            ax[0].plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[0].set_ylabel('r [mm]')
        ax[0].set_xlabel(r'$\phi $')

        ax[1].scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=1.)
            ax[1].plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[1].set_ylabel('r [mm]')
        ax[1].set_xlabel('z [mm]')
        
        plt.savefig(pdf_dir+'Cylindrical_initial_graph.pdf')

        print('Plotting: Ground Truth in Cylindrical coordinates!')

        fig, ax = plt.subplots(1, 2, figsize = (10,5),sharey=True, tight_layout=True)
        cmap = plt.get_cmap('bwr_r')

        ax[0].scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=float(y[j]))
            ax[0].plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[0].set_ylabel('r [mm]')
        ax[0].set_xlabel(r'$\phi $')
        
        ax[1].scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=float(y[j]))
            ax[1].plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[1].set_ylabel('r [mm]')
        ax[1].set_xlabel('z [mm]')

        plt.savefig(pdf_dir+'Cylindrical_truth_graphs.pdf')

        print('Plotting: Initial Graph colored in Cylindrical coordinates!')

        fig, ax = plt.subplots(1, 2, figsize = (10,5),sharey=True, tight_layout=True)
        cmap = plt.get_cmap('bwr_r')
        color = {0:'red',1:'blue'}

        ax[0].scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax[0].plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[0].set_ylabel('r [mm]')
        ax[0].set_xlabel(r'$\phi $')
    
        ax[1].scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax[1].plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[1].set_ylabel('r [mm]')
        ax[1].set_xlabel('z [mm]')

        plt.savefig(pdf_dir+'Cylindrical_initial_graph_colored.pdf')
    

def plot_cartesian(filenames,n_section,n_files):
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        theta_counter = 0#np.pi/n_section # start 45 degree rotated
        for i in range(n_files):
            print('Plotting file: ' + filenames[i*2])
            X, Ri, Ro, y = load_graph(filenames[i*2])
            #print('Zmin: %.2f, Zmax: %.2f' %(min(X[:,2]),max(X[:,2]))   )
            X[:,1] = X[:,1] * np.pi/n_section
            theta = (X[:,1] + theta_counter)%(np.pi*2)
           
            ax.scatter(1000*X[:,0]*np.cos(theta), 1000*X[:,0]*np.sin(theta), c='k')
            #ax1.scatter(1000*X[:,0]*np.cos(theta), 1000*X[:,2], c='k')

            feats_o = X[np.where(Ri.T)[1]]
            feats_i = X[np.where(Ro.T)[1]]

            x_o = 1000*feats_o[:,0]*np.cos(feats_o[:,1]+theta_counter)
            x_i = 1000*feats_i[:,0]*np.cos(feats_i[:,1]+theta_counter)
            y_o = 1000*feats_o[:,0]*np.sin(feats_o[:,1]+theta_counter)
            y_i = 1000*feats_i[:,0]*np.sin(feats_i[:,1]+theta_counter)

            for j in range(y.shape[0]):
                seg_args = dict(c='C'+str(i), alpha=float(y[j]*2))
                ax.plot([x_o[j],x_i[j]],[y_o[j],y_i[j]], '-', **seg_args)
            

            theta_counter += 2*np.pi/n_section
            theta_counter = theta_counter%(np.pi*2)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        #ax1.set_xlabel('$x [mm]$')
        #ax1.set_ylabel('$z [mm]$')
        #plt.tight_layout()
        plt.savefig(pdf_dir+'Cartesian.pdf')
        #plt.show()
def plot_3d(filenames,n_section,n_files):

    def change_view(az=20):
        ax.view_init(elev=10, azim=az%360)
        return ax 

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    cmap = plt.get_cmap('bwr_r')
    theta_counter = np.pi/n_section # start 45 degree rotated
    for i in range(n_files):
        print('Plotting file: ' + filenames[i])
        X, Ri, Ro, y = load_graph(filenames[i])
        X[:,1] = X[:,1] * np.pi/n_section
        theta = (X[:,1] + theta_counter)%(np.pi*2)
           
        ax.scatter(1000*X[:,2],1000*X[:,0]*np.cos(theta), 1000*X[:,0]*np.sin(theta), c='k',s=1)

        feats_o = X[np.where(Ri.T)[1]]
        feats_i = X[np.where(Ro.T)[1]]

        x_o = 1000*feats_o[:,0]*np.cos(feats_o[:,1]+theta_counter)
        x_i = 1000*feats_i[:,0]*np.cos(feats_i[:,1]+theta_counter)
        y_o = 1000*feats_o[:,0]*np.sin(feats_o[:,1]+theta_counter)
        y_i = 1000*feats_i[:,0]*np.sin(feats_i[:,1]+theta_counter)
        z_o = 1000*feats_o[:,2]
        z_i = 1000*feats_i[:,2]
        
        for j in range(y.shape[0]):
            seg_args = dict(c='C'+str(i), alpha=float(y[j]))
            ax.plot([z_o[j],z_i[j]],[x_o[j],x_i[j]],[y_o[j],y_i[j]],'-', **seg_args)
        
        if i%2==1:
            theta_counter += 2*np.pi/n_section
            theta_counter = theta_counter%(np.pi*2)
        
    ax.set_xlabel('$Z [mm]$')
    ax.set_ylabel('$X [mm]$')  
    ax.set_zlabel('$Y [mm]$') 
    ax.grid(b=None)
    change_view(45)
    ax.dist = 8 
    plt.savefig(pdf_dir+'Cartesian3D.pdf')
    #Make Gif
    """
    for angle in tqdm(range(360)):
        change_view(angle)
        plt.savefig('gif/Cartesian3D_%3d.pdf'%angle)
    """
    # Make Gif
    #anim = FuncAnimation(fig, change_view, frames=np.arange(0, 360), interval=100)
    #anim.save('Cartesian.gif', dpi=80)
def plot_combined(filenames,n_section):

        print('Plotting file: ' + filenames[1] + ' to: ' + pdf_dir)
        X, Ri, Ro, y = load_graph(filenames[1])

        feats_o = X[np.where(Ri.T)[1]]
        feats_i = X[np.where(Ro.T)[1]]
        
        feats_o[:,0] = feats_o[:,0]*1000 
        feats_o[:,1] = feats_o[:,1]*np.pi/n_section
        feats_o[:,2] = feats_o[:,2]*1000  
        feats_i[:,0] = feats_i[:,0]*1000  
        feats_i[:,1] = feats_i[:,1]*np.pi/n_section
        feats_i[:,2] = feats_i[:,2]*1000  
        
        print('Plotting: Initial Graph colored in Cylindrical and Cartesian coordinates!')

        fig, ax = plt.subplots(1, 3, figsize = (10,3),sharey=False, tight_layout=True)
        cmap = plt.get_cmap('bwr_r')
        color = {0:'red',1:'blue'}
        
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax[0].plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[0].scatter((np.pi/8)*X[:,1], 1000*X[:,0], s=10, c='k')
        ax[0].set_ylabel('r [mm]')
        ax[0].set_xlabel(r'$\phi$'+'\n\n (a)')
    
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax[1].plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[1].scatter(1000*X[:,2], 1000*X[:,0], s=10, c='k')
        ax[1].set_ylabel('r [mm]')
        ax[1].set_xlabel('z [mm]\n\n (b)')

        # Cartesian
        X[:,1] = X[:,1] * np.pi/n_section
        theta = X[:,1]%(np.pi*2)
           
        feats_o = X[np.where(Ri.T)[1]]
        feats_i = X[np.where(Ro.T)[1]]

        x_o = 1000*feats_o[:,0]*np.cos(feats_o[:,1])
        x_i = 1000*feats_i[:,0]*np.cos(feats_i[:,1])
        y_o = 1000*feats_o[:,0]*np.sin(feats_o[:,1])
        y_i = 1000*feats_i[:,0]*np.sin(feats_i[:,1])

        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax[2].plot([x_o[j],x_i[j]],[y_o[j],y_i[j]], '-', **seg_args)
        ax[2].scatter(1000*X[:,0]*np.cos(theta), 1000*X[:,0]*np.sin(theta), s=10, c='k')

        ax[2].set_xlabel('x [mm]\n\n (c)')
        ax[2].set_ylabel('y [mm]')

        plt.savefig(pdf_dir+'Initial_graph_colored_combined.pdf')
    

def main():
    input_dir = 'data/hitgraphs_big'
    n_section = 8
    n_files = 2
   
    input_dir = os.path.expandvars(input_dir)
    filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if f.startswith('event') and f.endswith('.npz')])
    filenames[:n_files] if n_files is not None else filenames
    
    #plot_3d(filenames,n_section,n_files)
    #plot_cartesian(filenames,n_section,1)
    #plot_cylindrical(filenames,n_section)
    plot_combined(filenames,n_section)

if __name__ == '__main__':
    pdf_dir = 'pdf/graphs/'
    main()
