import sys
import os
import cebra
sys.path.append("/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/shared_scripts")
import manifold_fit_and_decode_fns as mff
import fit_helper_fns as fhf
import numpy as np
import scipy.io
import cebra
from scipy import stats
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns; sns.set()
import timeit
from scipy.interpolate import splprep, splev
from scipy.spatial import distance_matrix

def nt_TDA(data, pct_distance=1, pct_neighbors=20):
  
    
    # Compute the pairwise distance matrix
    distances = distance_matrix(data, data)
    
    # Determine the neighborhood radius for each point based on the 1st percentile of distances
    neighborhood_radius = np.percentile(distances, pct_distance, axis=1)
    
    # Count the number of neighbors for each point within the neighborhood radius
    neighbor_counts = np.sum(distances <= neighborhood_radius[:, None], axis=1)
    
    # Identify points with a neighbor count below the 20th percentile
    threshold_neighbors = np.percentile(neighbor_counts, pct_neighbors)
    outlier_indices = np.where(neighbor_counts < threshold_neighbors)[0]
    
    # Remove outliers from the data
    cleaned_data = np.delete(data, outlier_indices, axis=0)
    
    return cleaned_data, outlier_indices
def fit_spud_to_cebra(embeddings_3d, nKnots=20, knot_order='wt_per_len', penalty_type='mult_len', length_penalty=5):
    # Set up the fit parameters, taken base from Chaudhuri et al.
    fit_params = {
        'dalpha': 0.005,
        'knot_order': knot_order,
        'penalty_type': penalty_type,
        'nKnots': nKnots,
        'length_penalty': length_penalty
    }

    # Create fitter object
    fitter = mff.PiecewiseLinearFit(embeddings_3d, fit_params)
    
    # Get initial knots
    unord_knots = fitter.get_new_initial_knots()
    init_knots = fitter.order_knots(unord_knots, method=fit_params['knot_order'])
    
    # Fit the data
    curr_fit_params = {'init_knots': init_knots, 'penalty_type': fit_params['penalty_type']}
    fitter.fit_data(curr_fit_params)
    
    # Get the final curve
    loop_final_knots = fhf.loop_knots(fitter.saved_knots[0]['knots'])
    tt, curve = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')
    
    return curve, tt
def plot_in_3d(embeddings,session, behav_var, name_behav_var,principal_curve=None):
    fig = plt.figure(figsize=(10, 8))
    
     # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    scatter_3d = ax2.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2])
    ax2.set_title(f"3D: Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
    ax2.set_xlabel('Embedding Dimension 1')
    ax2.set_ylabel('Embedding Dimension 2')
    ax2.set_zlabel('Embedding Dimension 3')


    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original points
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
                        c=behav_var, cmap='viridis', s=5)
    if(principal_curve is not None):
        # Plot the principal curve
        ax.plot(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2], color='red', linewidth=2)

    ax.set_xlabel('CEBRA Dimension 1')
    ax.set_ylabel('CEBRA Dimension 2')
    ax.set_zlabel('CEBRA Dimension 3')
    ax.set_title('Principal Curve of CEBRA-processed Neural Data')

    plt.colorbar(scatter, label=f'{name_behav_var}')
    plt.show()
# def create_rotating_3d_plot(embeddings_3d, session, behav_var, name_behav_var,anim_save_path, save_anim, principal_curve=None):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=behav_var, cmap='viridis', s=5)
#     if(principal_curve is not None):
#         ax.plot(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2], color='red', linewidth=2)
    
#     ax.set_title(f"3D: Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
#     ax.set_xlabel('Embedding Dimension 1')
#     ax.set_ylabel('Embedding Dimension 2')
#     ax.set_zlabel('Embedding Dimension 3')

#     plt.colorbar(scatter, label=f'{name_behav_var}')

#     def rotate(angle):
#         ax.view_init(elev=10., azim=angle)
#         return scatter,

#     anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50, blit=True)
    
#     if(anim):
#         if save_anim:
#             anim.save(f"{anim_save_path}_{name_behav_var}.gif", writer='pillow', fps=30)
#         else:
#             plt.show()

#     return anim

def create_rotating_3d_plot(embeddings_3d, session, behav_var, name_behav_var, anim_save_path, save_anim, principal_curve=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the static scatter points
    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=behav_var, cmap='viridis', s=5)
    
    # Plot the principal curve if provided
    if principal_curve is not None:
        ax.plot(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2], color='red', linewidth=2)

    # Add a dynamic point to indicate the trajectory moving over time
    dynamic_point, = ax.plot([], [], [], 'ro', markersize=8)  # 'ro' indicates a red point
    
    ax.set_title(f"3D: Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_zlabel('Embedding Dimension 3')

    plt.colorbar(scatter, label=f'{name_behav_var}')

    def update(frame):
        # Slow down the point's movement by updating every Nth frame
        point_index = frame // 10  # Adjust the divisor (10) to control speed, higher value = slower movement
        point_index = min(point_index, len(embeddings_3d) - 1)  # Ensure the index does not exceed data length

        # Update the moving point's position to match the current frame's embedding point
        dynamic_point.set_data([embeddings_3d[point_index, 0]], [embeddings_3d[point_index, 1]])  # Ensure the data is a sequence
        dynamic_point.set_3d_properties([embeddings_3d[point_index, 2]])  # Ensure the z-data is a sequence
        ax.view_init(elev=10., azim=frame)  # Rotate the view
        return dynamic_point, scatter

    # Create the animation, combining the rotation and moving trajectory point
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(embeddings_3d) * 10), interval=100, blit=True)

    if anim:
        if save_anim:
            anim.save(f"{anim_save_path}_{name_behav_var}.gif", writer='pillow', fps=30)
        else:
            plt.show()

    return anim

def apply_cebra(neural_data,output_dimensions,rm_outliers=True,max_iterations=None,batch_size=None):
    model = cebra.CEBRA(output_dimension=3, max_iterations=1000, batch_size=128)
    model.fit(neural_data)
    embeddings = model.transform(neural_data)
    print(embeddings.shape)
    if(rm_outliers):
        embeddings, outlier_indices = nt_TDA(embeddings)
    print(f"Output embeddings shape: {embeddings.shape}")
    return embeddings
def plot_in_2d(embeddings,session, behav_var, name_behav_var,principal_curve=None):
    fig, ax = plt.subplots(figsize=(10,8))
    # 2D plot
    scatter_2d = ax.scatter(embeddings[:, 0], embeddings[:, 1],c=behav_var, cmap='viridis', s=5)
    ax.set_title(f"2D: Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    plt.colorbar(scatter_2d, label=f'{name_behav_var}')
    if(principal_curve is not None):
        # Plot the principal curve
        ax.plot(principal_curve[:, 0], principal_curve[:, 1], color='red', linewidth=2)
    plt.show()

