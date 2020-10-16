import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from tools import *
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')

font = {
        'size'   : 16,
    }

axes = {
        'titlesize' : 16,
        'labelsize' : 16,
    }

matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)


def count_edges(filenames):

        total_false_edges = np.zeros(len(filenames))
        total_true_edges = np.zeros(len(filenames))

        for idx, file in enumerate(filenames):

            X, Ri, Ro, y = load_graph(file)

            total_true_edges[idx] = sum(y)
            total_false_edges[idx] = y.shape[0] - sum(y)


        print('The dataset contains:')
        print('All Edges:')
        print('Total: %d, Average: %4.1f +/- %3.1f'%(np.sum(total_true_edges+total_false_edges),np.mean(total_true_edges+total_false_edges),np.std(total_true_edges+total_false_edges) ))
        print('True Edges:')
        print('Total: %d, Average: %4.1f +/- %3.1f'%(np.sum(total_true_edges),np.mean(total_true_edges),np.std(total_true_edges) ))
        print('Fake Edges:')
        print('Total: %d, Average: %4.1f +/- %3.1f'%(np.sum(total_false_edges),np.mean(total_false_edges),np.std(total_false_edges) ))

def count_nodes(filenames):

        total_nodes = np.zeros(len(filenames))

        for idx, file in enumerate(filenames):

            X, Ri, Ro, y = load_graph(file)
            total_nodes[idx] = X.shape[0]

        print('The dataset contains:')
        print('Total Nodes:')
        print('Total: %d, Average: %4.1f +/- %3.1f'%(np.sum(total_nodes),np.mean(total_nodes),np.std(total_nodes)))

def find_edges(filenames):

    total_true_edges  = np.zeros((9, len(filenames)))
    total_false_edges = np.zeros((9, len(filenames)))

    for idx, file in enumerate(filenames):
        X, Ri, Ro, y = load_graph(file)

        edge_i = X[np.where(Ro.T)[1]]

        for i in range(edge_i.shape[0]):
            total_true_edges[find_layer(edge_i[i,0]),idx] +=  y[i] 
            total_false_edges[find_layer(edge_i[i,0]),idx] += 1-y[i] 

    print('Edges by layer:')
    for i in range(9):
        print('Layers: %d - %d'%(i+1,i+2))
        print('(True)  Total: %d, Average: %4.1f +/- %3.1f'%(np.sum(total_true_edges, axis=1)[i], np.mean(total_true_edges, axis=1)[i], np.std(total_true_edges, axis=1)[i]))
        print('(Fake) Total: %d, Average: %4.1f +/- %3.1f'%(np.sum(total_false_edges, axis=1)[i], np.mean(total_false_edges, axis=1)[i], np.std(total_false_edges, axis=1)[i]))


    fig, ax = plt.subplots(1, 1, tight_layout=True)
    layers  = ['1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10']
    n_true  = np.sum(total_true_edges, axis=1)
    n_false = np.sum(total_false_edges, axis=1)
    ax.bar(layers, n_true,  color='navy', label = 'True')
    ax.bar(layers, n_false, bottom=n_true, color='darkorange', label = 'Fake')
    ax.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
    plt.legend()
    ax.set_xlabel('Layer')
    ax.set_ylabel('Amount of edges')
    plt.savefig('pdf/graphs/edge_distribution.pdf')
    plt.savefig('png/graphs/edge_distribution.png')


def find_nodes(filenames):

    total_nodes = np.zeros((10, len(filenames)))

    for idx, file in enumerate(filenames):
        X, Ri, Ro, y = load_graph(file)

        for i in range(X.shape[0]):
            total_nodes[find_layer(X[i,0]),idx] += 1

    print('Nodes by layer:')
    for i in range(10):
        print('Layer: %d Total: %d, Average: %4.1f +/- %3.1f'%(i+1,np.sum(total_nodes, axis=1)[i], np.mean(total_nodes, axis=1)[i], np.std(total_nodes, axis=1)[i]))


    fig, ax = plt.subplots(1, 1, tight_layout=True)
    plt.xticks(np.arange(0, 11, step=1))
    layers  = range(1,11)
    n_nodes  = np.sum(total_nodes, axis=1)
    ax.bar(layers, n_nodes,  color='navy')
    ax.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Amount of nodes')
    plt.savefig('pdf/graphs/node_distribution.pdf')
    plt.savefig('png/graphs/node_distribution.png')


def find_layer(value):
    value = value*1000
    if value < 50: 
        layer = 0
    elif (value >= 50) and (value < 80):
        layer = 1
    elif (value >= 80) and (value < 150):
        layer = 2
    elif (value >= 150) and (value < 200):
        layer = 3 
    elif (value >= 200) and (value < 300):
        layer = 4
    elif (value >= 300) and (value < 400):
        layer = 5
    elif (value >= 400) and (value < 600):
        layer = 6
    elif (value >= 600) and (value < 750):
        layer = 7
    elif (value >= 750) and (value < 900):
        layer = 8
    elif (value >= 900) and (value < 1050):
        layer = 9
    else:
        raise ValueError()

    return layer




def main():
    train_dir = 'data/graph_data/train'
    valid_dir = 'data/graph_data/valid'
   
    train_dir = os.path.expandvars(train_dir)
    train_names = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) 
                if f.startswith('event') and f.endswith('.npz')])

    valid_dir = os.path.expandvars(valid_dir)
    valid_names = sorted([os.path.join(valid_dir, f) for f in os.listdir(valid_dir) 
                if f.startswith('event') and f.endswith('.npz')])

    filenames = train_names + valid_names


    print('The dataset contains:')
    print('%d subgraphs'%len(filenames))

    count_edges(filenames)
    count_nodes(filenames)
    find_nodes(filenames)
    find_edges(filenames)

if __name__ == '__main__':
    pdf_dir = 'pdf/graphs/'
    png_dir = 'png/graphs/'
    gif_dir = 'gif/'
    main()
