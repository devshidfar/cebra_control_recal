import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import ot  # POT: Python Optimal Transport
from scipy.io import savemat

plt.close('all')

def random_rotation():
    """Generate a random SO(3) rotation matrix."""
    return R.random().as_matrix()

def rotate_embeddings(embeddings, rotation_matrix):
    """Apply a given rotation matrix to the embeddings."""
    return embeddings @ rotation_matrix.T

def compute_emd_3d(source_cloud, target_cloud):
    """Compute true 3D Earth Mover's Distance (EMD) using Optimal Transport."""
    n = min(len(source_cloud), len(target_cloud))  # Match sizes if different
    # Cost matrix: pairwise Euclidean distances
    cost_matrix = np.linalg.norm(source_cloud[:n, None] - target_cloud[:n], axis=2)

    # Uniform weights 
    weights_source = np.ones(n) / n
    weights_target = np.ones(n) / n

    # Solve Optimal Transport problem
    emd_value = ot.emd2(weights_source, weights_target, cost_matrix)  # EMD squared

    return emd_value

def find_best_rotation(source_cloud, target_cloud, num_trials=100):
    """Find the rotation that minimizes 3D Earth Mover's Distance (EMD)."""
    best_rotation = None
    min_distance = float('inf')
    best_cloud = None

    for _ in range(num_trials):
        rotation = random_rotation()
        rotated_cloud = rotate_embeddings(source_cloud, rotation)
        distance = compute_emd_3d(rotated_cloud, target_cloud)

        if distance < min_distance:
            min_distance = distance
            best_rotation = rotation
            best_cloud = rotated_cloud

    return best_rotation, best_cloud, min_distance

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import plotly.graph_objects as go

def plot_point_clouds_interactive(clouds, session_num, save_dir, title="Point Cloud Visualization"):
    """
    Create an interactive 3D plot of multiple point clouds using Plotly and save it as an HTML file.
    
    Parameters:
        clouds (list of np.ndarray): A list of point clouds (each with shape (N, 3)).
        session_num (int): Identifier used to name the saved file.
        save_dir (str): Directory where the HTML file will be saved.
        title (str): Title of the plot.
    """
    # Create the Plotly figure
    fig = go.Figure()

    # Define colors for the traces
    colors = ['red', 'green', 'blue', 'magenta']
    
    for i, cloud in enumerate(clouds):
        fig.add_trace(go.Scatter3d(
            x=cloud[:, 0],
            y=cloud[:, 1],
            z=cloud[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors[i % len(colors)]),
            name=f"Cloud {i+1}"
        ))
    
    # Update layout with title and axis labels
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Define the file save path (saved as HTML)
    save_path = os.path.join(save_dir, f"session_{session_num}.html")
    
    # Save the interactive plot to an HTML file
    fig.write_html(save_path)
    print(f"Interactive plot saved to {save_path}")


def plot_point_clouds(clouds=None, session_num=None, save_dir=None, title="Point Cloud Visualization"):
    """Plot multiple 3D point clouds and save the plot as an image."""
    
    
    # Define the save path
    save_path = os.path.join(save_dir, f"session_{session_num}.png")

    # Create the figure and 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'm']
    for i, cloud in enumerate(clouds[:len(colors)]):
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=colors[i], alpha=0.7, label=f"Cloud {i+1}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    # Save the plot instead of showing it
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

    print(f"Plot saved to {save_path}")


# Load data and run
if __name__ == "__main__":
    # Load embeddings (adjust file path)
    file_path = '/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results/trial_type_test/trial_type_test_all_sessions_data.mat'
    mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    
    trials = [mat_data[f'full_trial_{i}'] for i in range(1, 5)]
    sessions = [trial.sessions for trial in trials]
    embeddings_list = [[session.embeddings_3d for session in s] for s in sessions]

    save_dir = "wasserstein_plots"
    os.makedirs(save_dir, exist_ok=True)
    num_trials = 4

    for k in range(2,10):
        target_cloud = embeddings_list[0][k]
        source_clouds = [embeddings_list[i][k] for i in range(1, num_trials)]

        # Find best rotation using true 3D EMD
        N = source_clouds[0].shape[0]
        best_rotation = np.zeros((num_trials-1,3,3))
        best_rotated_cloud = np.zeros((num_trials-1,N,3))
        min_distance = np.zeros(num_trials-1)

        for i in range(0,num_trials-1):
            best_rotation[i], best_rotated_cloud[i], min_distance[i] = find_best_rotation(source_clouds[i], target_cloud)

            # Plot original and rotated point clouds
            # plot_point_clouds([source_clouds[i], target_cloud])
            # plot_point_clouds([best_rotated_cloud[i], target_cloud])
            
        #plot_point_clouds(clouds=[best_rotated_cloud[0],best_rotated_cloud[1],best_rotated_cloud[2],target_cloud],session_num=k,save_dir=save_dir,title=f"Embeddings Same Trial {k}")
        plot_point_clouds_interactive(clouds=[best_rotated_cloud[0],best_rotated_cloud[1],best_rotated_cloud[2],target_cloud],session_num=k,save_dir=save_dir,title=f"Embeddings Same Trial {k+28}")
        print(f"Minimized Wasserstein (EMD) Distance in 3D: {min_distance}")


        modified_data = mat_data.copy
        output_file_path = os.path.join(file_path,'rotated')

        # for k in range(7,8):

        #     for i in range(1, num_trials):  # Skip target (session 0)
        #         session_key = f"full_trial_{i}"
                
        #         # Retrieve session and modify
        #         modified_data[session_key].sessions[k].rotated_embeddings_3d = best_rotated_cloud[i-1]

        # Save the modified data
        # savemat(output_file_path, modified_data)