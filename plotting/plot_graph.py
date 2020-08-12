import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tools import *
from tqdm import tqdm
from matplotlib.lines import Line2D

def plot_cylindrical(filenames,n_section):

        print('Plotting file: ' + filenames[0] + ' to: ' + pdf_dir)
        X, Ri, Ro, y = load_graph(filenames[0])

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
        ax[0].set_xlabel(r'$\phi$'+' [rad]')

        ax[1].scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=1.)
            ax[1].plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[1].set_ylabel('r [mm]')
        ax[1].set_xlabel('z [mm]')
        
        plt.savefig(pdf_dir+'Cylindrical_initial_graph.pdf')
        plt.savefig(png_dir+'Cylindrical_initial_graph.png')

        print('Plotting: Ground Truth in Cylindrical coordinates!')

        fig, ax = plt.subplots(1, 2, figsize = (10,5),sharey=True, tight_layout=True)
        cmap = plt.get_cmap('bwr_r')

        ax[0].scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=float(y[j]))
            ax[0].plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[0].set_ylabel('r [mm]')
        ax[0].set_xlabel(r'$\phi$'+' [rad]')
        
        ax[1].scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c='r', alpha=float(y[j]))
            ax[1].plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[1].set_ylabel('r [mm]')
        ax[1].set_xlabel('z [mm]')

        plt.savefig(pdf_dir+'Cylindrical_truth_graphs.pdf')
        plt.savefig(png_dir+'Cylindrical_truth_graphs.png')

        print('Plotting: Initial Graph colored in Cylindrical coordinates!')

        fig, ax = plt.subplots(1, 2, figsize = (10,5),sharey=True, tight_layout=True)
        cmap = plt.get_cmap('bwr_r')
        color = {0:'red',1:'blue'}

        ax[0].scatter((np.pi/8)*X[:,1], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax[0].plot([feats_o[j,1],feats_i[j,1]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[0].set_ylabel('r [mm]')
        ax[0].set_xlabel(r'$\phi$'+' [rad]')
    
        ax[1].scatter(1000*X[:,2], 1000*X[:,0], c='k')
        for j in range(y.shape[0]):
            seg_args = dict(c=color[int(y[j])], alpha=1.)
            ax[1].plot([feats_o[j,2],feats_i[j,2]],[feats_o[j,0],feats_i[j,0]], '-', **seg_args)
        ax[1].set_ylabel('r [mm]')
        ax[1].set_xlabel('z [mm]')
    
        # create custom legend
        legend_elements = [Line2D([0], [0], color='red', label='false'),
            Line2D([0], [0], color='blue', label='true')]

        ax[0].legend(handles=legend_elements)
        ax[1].legend(handles=legend_elements)

        plt.savefig(pdf_dir+'Cylindrical_initial_graph_colored.pdf')
        plt.savefig(png_dir+'Cylindrical_initial_graph_colored.png')
def plot_cartesian(filenames,n_section,n_files):
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        cmap = plt.get_cmap('bwr_r')
        theta_counter = 0 # start 45 degree rotated
        for i in range(n_files):
            print('Plotting file: ' + filenames[i])
            X, Ri, Ro, y = load_graph(filenames[i])
            #print('Zmin: %.2f, Zmax: %.2f' %(min(X[:,2]),max(X[:,2]))   )
            X[:,1] = X[:,1] * np.pi/n_section
            theta = (X[:,1] + theta_counter)%(np.pi*2)
           
            ax[0].scatter(1000*X[:,0]*np.cos(theta), 1000*X[:,0]*np.sin(theta), c='k', s=3)
            ax[1].scatter(1000*X[:,0]*np.cos(theta), 1000*X[:,0]*np.sin(theta), c='k', s=3)

            #ax1.scatter(1000*X[:,0]*np.cos(theta), 1000*X[:,2], c='k')

            feats_o = X[np.where(Ri.T)[1]]
            feats_i = X[np.where(Ro.T)[1]]

            x_o = 1000*feats_o[:,0]*np.cos(feats_o[:,1]+theta_counter)
            x_i = 1000*feats_i[:,0]*np.cos(feats_i[:,1]+theta_counter)
            y_o = 1000*feats_o[:,0]*np.sin(feats_o[:,1]+theta_counter)
            y_i = 1000*feats_i[:,0]*np.sin(feats_i[:,1]+theta_counter)

            # print all edges
            for j in range(y.shape[0]):
                seg_args = dict(c='darkblue')
                ax[0].plot([x_o[j],x_i[j]],[y_o[j],y_i[j]], '-', **seg_args)
            # print only true edges
            for j in range(y.shape[0]):
                seg_args = dict(c='darkblue', alpha=y[j])
                ax[1].plot([x_o[j],x_i[j]],[y_o[j],y_i[j]], '-', **seg_args)

            # draw seperator line
            ax[0].plot([0,1100*np.cos(theta_counter-np.pi/n_section)],[0,1100*np.sin(theta_counter-np.pi/n_section)],'-', c='darkorange')
            ax[0].plot([0,1100*np.cos(theta_counter+np.pi/n_section)],[0,1100*np.sin(theta_counter+np.pi/n_section)],'-', c='darkorange')
            ax[1].plot([0,1100*np.cos(theta_counter-np.pi/n_section)],[0,1100*np.sin(theta_counter-np.pi/n_section)],'-', c='darkorange')
            ax[1].plot([0,1100*np.cos(theta_counter+np.pi/n_section)],[0,1100*np.sin(theta_counter+np.pi/n_section)],'-', c='darkorange')


            theta_counter += 2*np.pi/n_section
            theta_counter = theta_counter%(np.pi*2)

        ax[0].set_xlabel('x [mm]')
        ax[0].set_ylabel('y [mm]')
        ax[1].set_xlabel('$x [mm]$')
        ax[1].set_ylabel('$y [mm]$')
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[0].set_title('All Edges (After Preprocessing)')
        ax[1].set_title('Only True Edges (After Preprocessing)')
        plt.savefig(pdf_dir+'Cartesian.pdf')
        plt.savefig(png_dir+'Cartesian.png')
        plt.tight_layout()

        print('Plot saved to: ' + pdf_dir+'Cartesian.pdf')
        #plt.show()
def plot_3d(filenames,n_section,n_files, make_gif=False):

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
    plt.savefig(png_dir+'Cartesian3D.png')
    if make_gif == True:
	    # Make Gif
	    rotation_angle = 360  # total rotation angle in degrees
	    duration = 	5 # duration of gif in seconds
	    anim = FuncAnimation(fig, change_view, frames=np.arange(0, rotation_angle), interval=duration*1000/rotation_angle)
	    anim.save(gif_dir+'Cartesian3D.gif', dpi=80, writer='PillowWriter')
	    
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
        plt.savefig(png_dir+'Initial_graph_colored_combined.png')
def main():
    input_dir = 'data/example_event/'
    n_section = 8
    n_files = 16
   
    input_dir = os.path.expandvars(input_dir)
    filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if f.startswith('event') and f.endswith('.npz')])
    filenames[:n_files] if n_files is not None else filenames

    plot_3d(filenames,n_section,n_files, make_gif=False)
    plot_cartesian([filenames[i*2] for i in range(n_files//2)],n_section,n_files//2)
    plot_cylindrical(filenames,n_section)
    plot_combined(filenames,n_section)

if __name__ == '__main__':
    pdf_dir = 'pdf/graphs/'
    png_dir = 'png/graphs/'
    gif_dir = 'gif/'
    main()
