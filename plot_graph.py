import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from datasets.hitgraphs import HitGraphDataset
import os,sys
from datasets.graph import load_graph
from tqdm import tqdm

def plot_cylindrical(filenames,n_section,n_files):

        print('Plotting file: ' + filenames[0] + ' to: ' + png_dir)
        X, Ri, Ro, y = load_graph(filenames[0])

        feats_o = X[np.where(Ri.T)[1]]
        feats_i = X[np.where(Ro.T)[1]]
        
        feats_o[:,0] = feats_o[:,0]*1000 
        feats_o[:,1] = feats_o[:,1]*np.pi/n_section
        feats_o[:,2] = feats_o[:,2]*1000  
        feats_i[:,0] = feats_i[:,0]*1000  
        feats_i[:,1] = feats_i[:,1]*np.pi/n_section
        feats_i[:,2] = feats_i[:,2]*1000  
        
        print('Plotting: Cylindrical_phi_AP.png')
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        ax.scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=1.)
            ax.plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax.set_ylabel('$R [mm]$')
        ax.set_xlabel('$\Phi $')
        plt.savefig(png_dir+'Cylindrical_phi_AP.png')

        print('Plotting: Cylindrical_phi_truth.png')
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        ax.scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=float(y[j]))
            ax.plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax.set_ylabel('$R [mm]$')
        ax.set_xlabel('$\Phi $')
        plt.savefig(png_dir+'Cylindrical_phi_truth.png')

        print('Plotting: Cylindrical_phi.png')
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        ax.scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        color = {0:'red',1:'blue'}
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax.plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax.set_ylabel('$R [mm]$')
        ax.set_xlabel('$\Phi $')
        plt.savefig(png_dir+'Cylindrical_phi.png')

        print('Plotting: Cylindrical_z_AP.png')
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        ax.scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=1.)
            ax.plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax.set_ylabel('$R [mm]$')
        ax.set_xlabel('$Z [mm] $')
        plt.savefig(png_dir+'Cylindrical_z_AP.png')

        print('Plotting: Cylindrical_z_truth.png')
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        ax.scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=float(y[j]))
            ax.plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax.set_ylabel('$R [mm]$')
        ax.set_xlabel('$Z [mm]$')
        plt.savefig(png_dir+'Cylindrical_z_truth.png')

        print('Plotting: Cylindrical_z.png')
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        ax.scatter(1000*X[:,2], 1000*X[:,0], c='k')
        color = {0:'red',1:'blue'}
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax.plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax.set_ylabel('$R [mm]$')
        ax.set_xlabel('$Z [mm]$')
        plt.savefig(png_dir+'Cylindrical_z.png')
    

def plot_cartesian(filenames,n_section,n_files):
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('bwr_r')
        theta_counter = np.pi/n_section # start 45 degree rotated
        for i in range(n_files//2):
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

        ax.set_xlabel('$X [mm]$')
        ax.set_ylabel('$Y [mm]$')
        #ax1.set_xlabel('$x [mm]$')
        #ax1.set_ylabel('$z [mm]$')
        #plt.tight_layout()
        plt.savefig(png_dir+'Cartesian.png')
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
    plt.savefig(png_dir+'Cartesian3D.png')
    #Make Gif
    """
    for angle in tqdm(range(360)):
        change_view(angle)
        plt.savefig('gif/Cartesian3D_%3d.png'%angle)
    """
    # Make Gif
    #anim = FuncAnimation(fig, change_view, frames=np.arange(0, 360), interval=100)
    #anim.save('Cartesian.gif', dpi=80)

    
    
def main():
    input_dir = 'data/hitgraphs_big'
    n_section = 8
    n_files = 16
   
    input_dir = os.path.expandvars(input_dir)
    filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if f.startswith('event') and f.endswith('.npz')])
    filenames[:n_files] if n_files is not None else filenames
    
    plot_cylindrical(filenames,n_section,n_files)



if __name__ == '__main__':
    png_dir = 'png/graphs/'
    main()
