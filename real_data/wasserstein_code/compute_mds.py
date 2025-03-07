import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.spatial.transform import Rotation as R
import ot  # POT: Python Optimal Transport
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import MDS

file_path = '/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results/all_rat_913/all_rat_913_all_sessions_data_backup.mat'
mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)

similarity_matrix = mat_data['pairwise_EMD_distance_matrix']
sessions = mat_data['sesions_for_EMD']
mean_H_error = mat_data['mean_H_error']

dimension = 2

if dimension == 2:
    n_components = 3
    mds = MDS(n_components=n_components, dissimilarity='precomputed')
    embeddings = mds.fit_transform(similarity_matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:,0], embeddings[:,1], color='blue', alpha=0.6, edgecolors=None)


    for i in range(len(embeddings[:,0])):
        # if(sessions[i] <= 34):
        #     plt.annotate(f"883,{sessions[i]}", (embeddings[i,0], embeddings[i,1]), textcoords="offset points", xytext=(5,5), ha='right',color='red')
        # else:
        #     plt.annotate(f"913,{sessions[i]}", (embeddings[i,0], embeddings[i,1]), textcoords="offset points", xytext=(5,5), ha='right',color='green')

        if(mean_H_error[i] <= 0.05):
            plt.annotate(f"H_err:{mean_H_error[i]:.3f},sess:{sessions[i]}", (embeddings[i,0], embeddings[i,1]), textcoords="offset points", xytext=(5,5), ha='right',color='green')
        else:
            plt.annotate(f"H_err:{mean_H_error[i]:.3f},sess:{sessions[i]}", (embeddings[i,0], embeddings[i,1]), textcoords="offset points", xytext=(5,5), ha='right',color='red')
    
    # Labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Scatter Plot')

    # Show the plot
    plt.show()

elif dimension == 3:
    n_components = 3
    mds = MDS(n_components=n_components, dissimilarity='precomputed')
    embeddings = mds.fit_transform(similarity_matrix)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot scatter points
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], color='blue', alpha=0.6, edgecolors=None)

    # Annotate points with different colors based on session number
    for i in range(len(embeddings[:, 0])):
        label = f"883,{sessions[i]}" if sessions[i] <= 34 else f"913,{sessions[i]}"
        color = 'red' if sessions[i] <= 34 else 'green'
        
        ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], 
                label, color=color, fontsize=8)

    # Labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D MDS Scatter Plot')

    # Show the plot
    plt.show()


print(embeddings.shape)

