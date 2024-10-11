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
    
def create_rotating_3d_plot(embeddings_3d, session, behav_var, name_behav_var,anim_save_path, save_anim, principal_curve=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=behav_var, cmap='viridis', s=5)
    if(principal_curve is not None):
        ax.plot(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2], color='red', linewidth=2)
    
    ax.set_title(f"3D: Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_zlabel('Embedding Dimension 3')

    plt.colorbar(scatter, label=f'{name_behav_var}')

    def rotate(angle):
        ax.view_init(elev=10., azim=angle)
        return scatter,

    anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50, blit=True)
    
    if(anim):
        if save_anim:
            anim.save(f"{anim_save_path}{name_behav_var}.gif", writer='pillow', fps=30)
        else:
            plt.show()

    return anim

def apply_cebra(neural_data,output_dimensions,rm_outliers=True,max_iterations=None,batch_size=None):
    model = cebra.CEBRA(output_dimension=output_dimensions, max_iterations=1000, batch_size=128)
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

def plot_embeddings_side_by_side(embeddings_2d, embeddings_3d, umap_embeddings, session, hipp_angle_binned, true_angle_binned, principal_curve_2d, principal_curve_3d, save_path):
    """
    Plot CEBRA 2D, CEBRA 3D (projected to 2D), and UMAP 2D embeddings side by side.
    
    Parameters:
    - embeddings_2d (np.ndarray): CEBRA 2D embeddings.
    - embeddings_3d (np.ndarray): CEBRA 3D embeddings.
    - umap_embeddings (np.ndarray): UMAP 2D embeddings.
    - session: Session object containing metadata.
    - hipp_angle_binned (np.ndarray): Binned hippocampal angles.
    - true_angle_binned (np.ndarray): Binned true angles.
    - principal_curve_2d (np.ndarray): Principal curve for CEBRA 2D.
    - principal_curve_3d (np.ndarray): Principal curve for CEBRA 3D.
    - save_path (str): Path to save the figure.
    
    Returns:
    - None
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # CEBRA 2D Embedding
    sc1 = axes[0].scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=hipp_angle_binned % 360, cmap='viridis', s=10)
    axes[0].plot(principal_curve_2d[:,0], principal_curve_2d[:,1], color='red', linewidth=2)
    axes[0].set_title('CEBRA 2D Embedding')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    plt.colorbar(sc1, ax=axes[0], label='Hipp Angle (°)')

    # CEBRA 3D Embedding projected to 2D
    sc2 = axes[1].scatter(embeddings_3d[:,0], embeddings_3d[:,1], c=true_angle_binned % 360, cmap='plasma', s=10)
    axes[1].plot(principal_curve_3d[:,0], principal_curve_3d[:,1], color='red', linewidth=2)
    axes[1].set_title('CEBRA 3D Embedding (Projected to 2D)')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    plt.colorbar(sc2, ax=axes[1], label='True Angle (°)')

    # UMAP 2D Embedding
    sc3 = axes[2].scatter(umap_embeddings[:,0], umap_embeddings[:,1], c=true_angle_binned % 360, cmap='inferno', s=10)
    axes[2].set_title('UMAP 2D Embedding')
    axes[2].set_xlabel('UMAP Dimension 1')
    axes[2].set_ylabel('UMAP Dimension 2')
    plt.colorbar(sc3, ax=axes[2], label='True Angle (°)')

    plt.suptitle(f"Rat {session.rat}, Day {session.day}, Epoch {session.epoch}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved embedding plots to {save_path}")

