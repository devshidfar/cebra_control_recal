import sys
import cebra
sys.path.append("/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/spud_code/shared_scripts")
import manifold_fit_and_decode_fns_custom as mff
import fit_helper_fns_custom as fhf
import numpy as np
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns; sns.set()
import timeit
from scipy.interpolate import splprep, splev, interp1d
from scipy.spatial import distance_matrix, KDTree
from sklearn.neighbors import LocalOutlierFactor
from ripser import ripser
from persim import plot_diagrams
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
sys.path.append('/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/SI_code')
from real_data.SI_code.structure_index import compute_structure_index, draw_graph
import os
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import os
import plotly.graph_objs as go
from plotly.offline import plot
from matplotlib.widgets import Slider
import mpld3
from scipy.signal import butter, filtfilt

def calculate_average_difference_in_decoded_hipp_angle(embeddings=None, principal_curve=None, tt=None, actual_angles=None,true_angles=None):
    """
    Calculate the average difference between the predicted hippocampal angles
    decoded from the spline and the actual hippocampal angles.

    Parameters:
    - embeddings (np.ndarray): Embedding points (num_samples, embedding_dimension).
    - principal_curve (np.ndarray): Principal curve (spline) fitted to the embeddings.
    - tt (np.ndarray): Parameterization along the principal curve.
    - actual_angles (np.ndarray): Actual hippocampal angles corresponding to embeddings.

    Returns:
    - avg_diff (float): Average absolute angular difference between decoded and actual angles.
    """

    # Decode angles from embeddings using the spline
    decoded_angles, mse = mff.decode_from_passed_fit(embeddings, tt[:-1], principal_curve[:-1], actual_angles)
    decoded_angles = decoded_angles + true_angles[3]

    angle_diff = abs(decoded_angles - actual_angles)

    # # Calculate average angular difference
    # avg_diff = calculate_average_angle_difference(decoded_angles, actual_angles)

    return angle_diff, decoded_angles, mse

def smooth_derivative(data=None, window_size=3):
    # Step 1: Compute finite differences
    diffs = np.diff(data)
    
    # Step 2: Compute the moving average of differences
    # Create a convolution kernel for averaging
    kernel = np.ones(window_size) / window_size
    avg_diffs = np.convolve(diffs, kernel, mode='valid')  # Valid ensures proper windowing
    
    return avg_diffs



def run_persistent_homology(embeddings, session_idx, session, results_save_path, dimension):
    """
    Runs persistent homology on embeddings, saves Betti number plots,
    and parametrizes embeddings in polar coordinates.

    Parameters:
    - embeddings (np.ndarray): Embedding points (num_samples, embedding_dimension).
    - session_idx (int): Index of the session.
    - session: Session object containing metadata.
    - results_save_path (str): Path to the results folder.
    - dimension (int): Dimension of the embeddings (e.g., 2 or 3).

    Saves:
    - Betti number plot (barcode diagram) to the session folder.
    - Polar coordinates of the embeddings (if computed).
    """


    # Create a folder for the session and dimension
    session_folder = os.path.join(results_save_path,"persistent_homology", f"session_{session_idx}", f"dimension_{dimension}")
    os.makedirs(session_folder, exist_ok=True)
    
    # Compute persistent homology
    ripser_result = ripser(embeddings, maxdim=1)
    diagrams = ripser_result['dgms']
    
    # Save the persistent homology results manually (as the shapes are inconsistent)
    homology_filename = os.path.join(session_folder, f"ripser_result_dimension_{dimension}.npz")
    
    # Save individual diagrams as arrays
    np.savez(homology_filename, H0=diagrams[0], est_H=diagrams[1])
    print(f"Saved persistent homology results (H0 and est_H) to {homology_filename}")
    
    # Plot the persistence diagrams
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_diagrams(diagrams, show=False, ax=ax)
    ax.set_title(f"Persistence Diagrams\nSession {session_idx} - Dimension {dimension}")
    plot_filename = os.path.join(session_folder, f"persistence_diagram_dimension_{dimension}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved persistence diagram plot to {plot_filename}")
    
    # Compute Betti numbers (number of persistent features in each dimension)
    betti_numbers = [len(diagrams[i]) for i in range(len(diagrams))]
    print(f"Betti numbers for session {session_idx}, dimension {dimension}:")
    for i, betti in enumerate(betti_numbers):
        print(f"  Betti_{i} = {betti}")
    
    # Save Betti numbers to a file
    betti_filename = os.path.join(session_folder, f"betti_numbers_dimension_{dimension}.txt")
    with open(betti_filename, 'w') as f:
        for i, betti in enumerate(betti_numbers):
            f.write(f"Betti_{i} = {betti}\n")
    print(f"Saved Betti numbers to {betti_filename}")
    
    # Parametrize the embeddings in polar coordinates
    polar_coords = None
    if dimension == 2:
        # Convert 2D embeddings to polar coordinates
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        theta = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        polar_coords = np.column_stack((theta, r))
        # Save the polar coordinates
        polar_filename = os.path.join(session_folder, f"polar_coordinates_dimension_{dimension}.npy")
        np.save(polar_filename, polar_coords)
        print(f"Saved polar coordinates to {polar_filename}")
    else:
        # For higher dimensions, compute circular coordinates using persistent cohomology
        try:
            from dreimac import CircularCoords
            # Parameters for Circular Coordinates
            prime = 47  # A prime number for the coefficient field
            n_landmarks = 1000
            cc = CircularCoords(embeddings, n_landmarks=n_landmarks, prime=prime)
            # Get circular coordinates
            theta = cc.get_coordinates()
            polar_coords = theta
            # Save the circular coordinates
            polar_filename = os.path.join(session_folder, f"circular_coordinates_dimension_{dimension}.npy")
            np.save(polar_filename, polar_coords)
            print(f"Saved circular coordinates to {polar_filename}")
        except ImportError:
            print("DREiMac library not installed. Cannot compute circular coordinates for dimension > 2.")
            print("Please install DREiMac to compute circular coordinates.")
    
    # Optionally, plot the parametrization
    if polar_coords is not None:
        plt.figure(figsize=(8, 6))
        if dimension == 2:
            plt.scatter(polar_coords[:, 0], polar_coords[:, 1], c='blue', s=5)
            plt.xlabel('Theta')
            plt.ylabel('Radius')
            plt.title(f'Polar Coordinates\nSession {session_idx} - Dimension {dimension}')
        else:
            plt.scatter(theta, np.zeros_like(theta), c='blue', s=5)
            plt.xlabel('Theta')
            plt.title(f'Circular Coordinates\nSession {session_idx} - Dimension {dimension}')
        param_plot_filename = os.path.join(session_folder, f"parametrization_dimension_{dimension}.png")
        plt.savefig(param_plot_filename)
        plt.close()
        print(f"Saved parametrization plot to {param_plot_filename}")
    else:
        print("Polar coordinates not computed.")

# def nt_TDA(data): #taken from Sebastien et al.

#     # Compute pairwise distances between points in the data
#     D = pairwise_distances(data)
#     print("Pairwise distances matrix:\n", D)  # Print the distance matrix
    
#     # Set the diagonal of the distance matrix to NaN to ignore self-distances
#     np.fill_diagonal(D, np.nan)
#     print("Distance matrix with NaN on diagonal:\n", D)  # Print modified distance matrix
    
#     # Calculate number of neighbors within the 5th percentile distance for each point
#     nn_dist = np.sum(D < np.nanpercentile(D, 5), axis=1)
#     print("Number of neighbors within 5th percentile:\n", nn_dist)  # Print neighbor count per point
    
#     # Identify noisy points where neighbor count is below the 20th percentile
#     noiseIdx = nn_dist < np.percentile(nn_dist, 20)
#     print("Noise index (True indicates noisy point):\n", noiseIdx)  # Print noise index array
    
#     # Print the sum of noisy points
#     print("Total number of noisy points:", sum(noiseIdx))  # Print count of noisy points
    
#     return noiseIdx


def nt_TDA(data, pct_distance=1, pct_neighbors=20,pct_dist=90):
    # Compute the pairwise distance matrix
    distances = distance_matrix(data, data)
    np.fill_diagonal(distances, 10)
    print("Pairwise distance matrix computed.")
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Sample distances (first 5 rows, first 5 columns):\n{distances[:5, :5]}\n")
    
    # Determine the neighborhood radius for each point based on the pct_distance percentile of distances
    neighborhood_radius = np.percentile(distances, pct_distance, axis=0)
    print(f"Neighborhood radius calculated using the {pct_distance}th percentile.")
    print(f"Neighborhood radius statistics:")
    print(f"  Min: {neighborhood_radius.min()}")
    print(f"  Max: {neighborhood_radius.max()}")
    print(f"  Mean: {neighborhood_radius.mean():.4f}")
    print(f"  Median: {np.median(neighborhood_radius):.4f}\n")
    
    # Count the number of neighbors for each point within the neighborhood radius
    neighbor_counts = np.sum(distances <= neighborhood_radius[:, None], axis=1)
    print("Neighbor counts computed for each point.")
    print(f"Neighbor counts statistics:")
    print(f"  Min: {neighbor_counts.min()}")
    print(f"  Max: {neighbor_counts.max()}")
    print(f"  Mean: {neighbor_counts.mean():.2f}")
    print(f"  Median: {np.median(neighbor_counts)}")
    print(f"  Example neighbor counts (first 10 points): {neighbor_counts[:50]}\n")
    
    # Identify points with a neighbor count below the pct_neighbors percentile
    threshold_neighbors = np.percentile(neighbor_counts, pct_neighbors)
    print(f"Threshold for neighbor counts set at the {pct_neighbors}th percentile: {threshold_neighbors}")
    
    outlier_indices = np.where(neighbor_counts < threshold_neighbors)[0]
    print(f"Outliers based on neighbor counts (count={len(outlier_indices)}): {outlier_indices}\n")
    
    # consider points as outliers if they are too far from any other points
    neighbgraph = NearestNeighbors(n_neighbors=5).fit(distances)
    dists, inds = neighbgraph.kneighbors(distances)
    min_distance_to_any_point = np.mean(dists, axis=1)
    print("Minimum distance to any other point calculated for each point.")


    print(f"Minimum distance statistics:")
    print(f"  Min: {min_distance_to_any_point.min()}")
    print(f"  Max: {min_distance_to_any_point.max()}")
    print(f"  Mean: {min_distance_to_any_point.mean():.4f}")
    print(f"  Median: {np.median(min_distance_to_any_point):.4f}\n")
    
    distance_threshold = np.percentile(min_distance_to_any_point, pct_dist)
    print(f"Distance threshold set at the {pct_dist}th percentile: {distance_threshold}")
    
    far_outliers = np.where(min_distance_to_any_point > distance_threshold)[0]
    print(f"Outliers based on distance threshold (count={len(far_outliers)}): {far_outliers}\n")
    
    # Combine with other outliers
    outlier_indices = np.unique(np.concatenate([outlier_indices, far_outliers]))
    print(f"Total outliers detected after combining criteria (count={len(outlier_indices)}): {outlier_indices}\n")
    
    # Compute inlier indices as all indices not in outlier_indices
    all_indices = np.arange(data.shape[0])
    inlier_indices = np.setdiff1d(all_indices, outlier_indices)
    print(f"Total inliers detected (count={len(inlier_indices)}): {inlier_indices}\n")
    
    # # Remove outliers from the data
    # cleaned_data = np.delete(data, outlier_indices, axis=0)
    
    return inlier_indices

def plot_time_vs_distance(embeddings, principal_curve, times,x_axis_var, annotate_var, annotate_var_name,session,session_idx, bin_size, dimension, save_path=None):
    """
    Plots time vs. distance of each embedding point to the spline with a velocity color map.

    Parameters:
    - embeddings (np.ndarray): Array of embedding points (num_bins, dim).
    - principal_curve (np.ndarray): Array of spline points (num_bins, dim).
    - times (np.ndarray): Array of time points corresponding to each embedding (num_bins,).
    - annotate_var (np.ndarray): Array of velocity values corresponding to each embedding (num_bins,).
    - session: Session object (for labeling purposes).
    - bin_size (int): Bin size in seconds (for labeling purposes).
    - save_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    # Compute Euclidean distances between embeddings and the spline
    distances = compute_min_distances(embeddings, principal_curve)
    save_path = f"{save_path}/session_{session_idx}/dimension_{dimension}"

    # Create the plot
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(times, distances, c=annotate_var, cmap='viridis', s=50, edgecolor='k', alpha=0.7)
    plt.xlabel(f'{x_axis_var}', fontsize=14)
    plt.ylabel('Distance to Spline', fontsize=14)
    plt.title(f'Session_idx: {session_idx}, Rat: {session.rat}, Day: {session.day}, Epoch: {session.epoch}, Dimension: {dimension}',  fontsize=16)
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{annotate_var_name}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        file_save_path = f"{save_path}/anno_var_{annotate_var_name}_x_ax_{x_axis_var}"
        os.makedirs(file_save_path, exist_ok=True)
        plt.savefig(file_save_path, dpi=300)
        plt.close()
        print(f"Saved time vs. distance plot to {save_path}")
    else:
        plt.show()

def plot_initial_knots(data_points, init_knots, session_idx, session, save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2],
               c='gray', s=5, alpha=0.5, label='Data Points')

    # Plot initial knots
    ax.scatter(init_knots[:, 0], init_knots[:, 1], init_knots[:, 2],
               c='red', s=100, marker='^', label='Initial Knots')

    ax.set_title(f'Initial Knots - Session {session_idx}\nRat {session.rat}, Day {session.day}, Epoch {session.epoch}')
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_zlabel('Embedding Dimension 3')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved initial knots plot to {save_path}")
    else:
        plt.show()


def fit_spud_to_cebra(embeddings, ref_angle=None,session_idx=None,
                       session=None, results_save_path=None,fit_params=None,dimension_3d=None,verbose=False):
#     # Set up the fit parameters, taken base from Chaudhuri et al.

    # Create fitter object
    fitter = mff.PiecewiseLinearFit(embeddings, fit_params)
    # Get initial knots
    unord_knots = fitter.get_new_initial_knots(method = 'kmedoids')
   
    init_knots = fitter.order_knots(unord_knots, method=fit_params['knot_order'])

    ######

    # **Plot the initial knots**
    # Define the save path for the plot
    if dimension_3d == 1:
        if results_save_path:
            plot_save_path = os.path.join(results_save_path, f"initial_knots_session_{session_idx}.png")
        else:
            plot_save_path = None  # Set to None to display the plot instead of saving

        # Call the plotting function
        plot_initial_knots(embeddings, init_knots, session_idx, session, save_path=plot_save_path)

    ######
    
    # Fit the data
    curr_fit_params = {'init_knots': init_knots, **fit_params}
    print(f"curr_fit_params: {curr_fit_params}")
    fitter.fit_data(curr_fit_params,verbose=verbose)

    # Get the final knots
    final_knots = fitter.saved_knots[0]['knots']
    final_knots_pre = final_knots
    print(f"final knots: {final_knots}")
    
    segments = np.vstack((final_knots[1:] - final_knots[:-1], final_knots[0] - final_knots[-1]))
    knot_dists = np.linalg.norm(segments, axis=1)
    print(f"knot_dists: {knot_dists}")
    max_dist = np.max(knot_dists)
    max_dist_idx = np.argmax(knot_dists)
    nKnots = final_knots.shape[0]
    if max_dist_idx < nKnots - 1:
        idx1 = max_dist_idx
        idx2 = max_dist_idx + 1
    else:
        # The segment is between the last knot and the first knot
        idx1 = nKnots - 1
        idx2 = 0

    print(f"The largest distance between consecutive knots is {max_dist}, "
          f"between knot {max_dist_idx} and knot {max_dist_idx + 1}")

    knot1 = final_knots[idx1]
    knot2 = final_knots[idx2]
    print(f"Knot {max_dist_idx}: {knot1}")
    print(f"Knot {max_dist_idx + 1}: {knot2}")

    
    if max_dist > 1.5*np.median(knot_dists):
        print(f"Removing outlier knot at index {max_dist_idx} with distance {max_dist}")
        # Remove the knot
        final_knots = np.delete(final_knots, idx2, axis=0)
    else:
        print("No outlier knot found; proceeding without removing any knots.")

    # Adjust the number of knots
    nKnots = final_knots.shape[0]

    # construct the spline
    loop_final_knots = fhf.loop_knots(final_knots)
    # loop_final_knots_pre = fhf.loop_knots(final_knots_pre)
    tt, curve = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')
    _, curve_pre = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')

    print(f"form of tt in fit_spud_to_cebra: {np.max(tt)-np.min(tt)}")
    print(f"form of hippangle in fit_spud_to_cebra: {np.max(ref_angle)-np.min(ref_angle)}")

    if ref_angle is not None:
        # Find the index of the closest hippocampal angle to the desired origin
        #origin_idx = np.argmin(np.abs(ref_angle - hippocampal_angle_origin))

        tt = tt * 2*(np.pi)

        # Shift the tt values so that the origin is aligned with tt = 0
        print(f"ref_angle: {ref_angle}")
        tt_shifted = tt
        tt_shifted = (tt + ref_angle) % (2*np.pi) # Keep tt in the [0, 2*pi] range
        print(f"first 10 tts: {tt[:10]}")
        print(f"first 10 tt shifteds: {tt_shifted[:10]}")
        print(f"First 100 tts: {tt[:100]}")

        # tt_diff = np.diff(tt_shifted)

        # # Check if the signs of the slopes match, if not, reverse the tt and curve
        # if np.sign(tt_diff[0]) != np.sign(angle_diff[0]):
        #     print("Reversing spline direction to align with hippocampal angles")
        #     tt_shifted = np.flip(tt_shifted)
        #     curve = np.flip(curve, axis=0)
        #     curve_pre = np.flip(curve_pre, axis=0)
        
        
        tt = tt_shifted
    
    return curve, curve_pre, tt


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
    

def create_rotating_3d_plot(embeddings_3d=None, session=None, behav_var=None, name_behav_var=None, anim_save_path=None, save_anim=None, principal_curve=None, tt=None, num_labels=10, mean_dist=None, avg_angle_diff=None, shuffled_avg_angle_diff=None):
    """
    Plots a 3D rotating plot of embeddings with the same color map for both `behav_var` and `tt` on the spline.
    Labels a certain number of points evenly spaced along the spline.

    Parameters:
    - embeddings_3d: 3D embeddings to plot.
    - session: Session metadata (for title).
    - behav_var: Behavioral variable to color-code the points.
    - name_behav_var: Name of the behavioral variable (for labeling).
    - anim_save_path: Path to save the animation.
    - save_anim: Whether to save the animation as a gif.
    - principal_curve: The spline fitted to the embeddings.
    - tt: Parametrization along the spline (optional, for coloring the spline).
    - num_labels: Number of evenly spaced labels to add along the spline.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    print(f"Range of behav_var: {np.min(behav_var) - np.max(behav_var)}")
    print(f"Range of parameter: {np.max(tt) - np.min(tt)}")
    print("Number of NaNs in behav_var:", np.sum(np.isnan(behav_var)))
    print("Number of Infs in behav_var:", np.sum(np.isinf(behav_var)))
    print("Number of NaNs in tt:", np.sum(np.isnan(tt)))
    print("Number of Infs in tt:", np.sum(np.isinf(tt)))

    # valid_indices = ~np.isnan(behav_var) & ~np.isnan(embeddings_3d).any(axis=1)
    # behav_var = behav_var[valid_indices]
    # embeddings_3d = embeddings_3d[valid_indices]

    # # Normalize `behav_var` and `tt` to use the same color scale
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=np.min(tt), vmax=np.max(tt))

    # Scatter plot of embeddings using behav_var as color map
    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                         c=behav_var, cmap=cmap, s=5)
    plt.colorbar(scatter, label=f'{name_behav_var}')

    #Plot the principal curve with the same color map, colored by `tt` values
    if principal_curve is not None and tt is not None:
        for i in range(len(principal_curve) - 1):
            color = cmap(norm(tt[i]))  # Assign color based on normalized tt
            ax.plot(principal_curve[i:i+2, 0], principal_curve[i:i+2, 1], principal_curve[i:i+2, 2], 
                    color=color, linewidth=2)

        # Add labels at evenly spaced points along the spline
        label_indices = np.linspace(0, len(principal_curve) - 1, num_labels, dtype=int)
        for idx in label_indices:
            x, y, z = principal_curve[idx]
            ax.text(x, y, z, f'tt={tt[idx]:.2f}', color='black', fontsize=8)

    #add mean dist
    if mean_dist is not None:
        # Position: (x, y) in figure coordinates [0,1]
        fig.text(
            0.05, 0.95,
            f'Mean Distance from spline: {mean_dist:.2f}, Avg angle diff: {avg_angle_diff:.2f} ({avg_angle_diff * (360/(2*np.pi)):.2f} degrees), Shufled avg angle diff: {shuffled_avg_angle_diff:.2f} ({shuffled_avg_angle_diff * (360/(2*np.pi)):.2f} degrees)',
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=5)
        )

    ax.set_title(f"3D: Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_zlabel('Embedding Dimension 3')

    # Rotation function for animation
    def rotate(angle):
        ax.view_init(elev=10., azim=angle)
        return scatter,

    anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50, blit=True)

    # Save or show the animation
    if anim:
        if save_anim:
            anim.save(f"{anim_save_path}{name_behav_var}.gif", writer='pillow', fps=30)
        else:
            plt.show()

    return anim

# def apply_cebra(neural_data=None,model_architecture='offset10-model',
#                         batch_size=512,
#                         learning_rate=3e-4,
#                         temperature_mode = 'constant',
#                         temperature=1,
#                         min_temperature = 1e-1,
#                         output_dimension=3,
#                         max_iterations=5000,
#                         distance='cosine',
#                         device='cuda_if_available',
#                         verbose=True):
    
#     model = cebra.CEBRA(model_architecture=model_architecture,
#                         batch_size=batch_size,
#                         learning_rate=learning_rate,
#                         temperature_mode = temperature_mode,
#                         temperature=temperature,
#                         min_temperature = min_temperature,
#                         output_dimension=output_dimension,
#                         max_iterations=max_iterations,
#                         distance=distance,
#                         device=device,
#                         verbose=verbose)
def apply_cebra(neural_data=None,output_dimension=3):

    ''' default hyper-params for CEBRA model

    # Model Architecture (model_architecture): 'offset10-model'
    # Batch Size (batch_size): 512
    # Temperature Mode (temperature_mode): "auto"
    # Learning Rate (learning_rate): 0.001
    # Max Iterations (max_iterations): 10,000
    # Time Offsets (time_offsets): 10
    # Output Dimension (output_dimension): 8
    # Device (device): "cuda_if_available" (falls back to "cpu" if no GPU is available)
    # Verbose (verbose): False

    '''
     
    model = cebra.CEBRA(output_dimension=output_dimension, max_iterations=1000, batch_size = 512)   
    model.fit(neural_data)
    embeddings = model.transform(neural_data)
    return embeddings

def dist_tot_to_princ_curve(embeddings=None,principal_curve=None):
        # Compute distances from embeddings to the principal curve
        distances = compute_min_distances(embeddings, principal_curve)

        # Calculate the mean distance
        mean_distance = np.mean(distances)

        print(f"The mean distance from the embeddings to the principal curve is: {mean_distance}")
        print(f"The min distance from the embeddings to the principal curve is: {np.min(distances)}")
        print(f"The max distance from the embeddings to the principal curve is: {np.max(distances)}")

        return mean_distance

def plot_in_2d(embeddings=None,session=None, behav_var=None, name_behav_var=None,principal_curve=None):
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

def compute_min_distances(embeddings, principal_curve):
    """
    Computes the minimum Euclidean distance from each embedding point to the principal curve.

    Parameters:
    - embeddings (np.ndarray): Array of embedding points (num_embeddings, dim).
    - principal_curve (np.ndarray): Array of spline points (num_spline_points, dim).

    Returns:
    - distances (np.ndarray): Array of minimum distances (num_embeddings,).
    """
    # Build a KDTree for the principal curve
    tree = KDTree(principal_curve)

    # Query the nearest distance for each embedding point
    distances, _ = tree.query(embeddings, k=1)  # k=1 for the nearest neighbor

    return distances

def interpolate_principal_curve(principal_curve, points_per_unit_distance=10):
    """
    Linearly interpolates between each pair of consecutive points in the principal curve.
    The number of interpolated points between two knots is proportional to the distance between them.

    Parameters:
    - principal_curve (np.ndarray): Array of spline points (num_knots, dim).
    - points_per_unit_distance (int): Number of interpolated points per unit distance.

    Returns:
    - interpolated_curve (np.ndarray): Finely interpolated spline points.
    """
    interpolated_curve = []

    for i in range(len(principal_curve) - 1):
        point_start = principal_curve[i]
        point_end = principal_curve[i + 1]
        segment = point_end - point_start
        distance = np.linalg.norm(segment)
        
        # Determine number of points to interpolate based on distance
        num_points = max(int(points_per_unit_distance * distance), 2)  # At least 2 points
        
        # Generate linearly spaced points between start and end
        interp_points = np.linspace(point_start, point_end, num=num_points, endpoint=False)
        interpolated_curve.append(interp_points)
    
    # Append the last point
    interpolated_curve.append(principal_curve[-1].reshape(1, -1))
    
    # Concatenate all interpolated segments
    interpolated_curve = np.vstack(interpolated_curve)
    
    return interpolated_curve

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
    sc1 = axes[0].scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=hipp_angle_binned, cmap='viridis', s=10)
    axes[0].plot(principal_curve_2d[:,0], principal_curve_2d[:,1], color='red', linewidth=2)
    axes[0].set_title('CEBRA 2D Embedding')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    plt.colorbar(sc1, ax=axes[0], label='Hipp Angle (°)')

    # CEBRA 3D Embedding projected to 2D
    sc2 = axes[1].scatter(embeddings_3d[:,0], embeddings_3d[:,1], c=true_angle_binned, cmap='plasma', s=10)
    axes[1].plot(principal_curve_3d[:,0], principal_curve_3d[:,1], color='red', linewidth=2)
    axes[1].set_title('CEBRA 3D Embedding (Projected to 2D)')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    plt.colorbar(sc2, ax=axes[1], label='True Angle (°)')

    # UMAP 2D Embedding
    sc3 = axes[2].scatter(umap_embeddings[:,0], umap_embeddings[:,1], c=true_angle_binned, cmap='inferno', s=10)
    axes[2].set_title('UMAP 2D Embedding')
    axes[2].set_xlabel('UMAP Dimension 1')
    axes[2].set_ylabel('UMAP Dimension 2')
    plt.colorbar(sc3, ax=axes[2], label='True Angle (°)')

    plt.suptitle(f"Rat {session.rat}, Day {session.day}, Epoch {session.epoch}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved embedding plots to {save_path}")
    
    return


def umap_and_tSNE_vis(neural_data,embeddings_2d,embeddings_3d,hipp_angle_binned,true_angle_binned,principal_curve_2d,principal_curve_3d,session,session_idx,results_save_path):
    # Compute UMAP embeddings
    print("Computing UMAP embeddings...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(neural_data)
    print(f"UMAP embeddings shape: {umap_embeddings.shape}")

    # Plot UMAP embeddings side by side
    plot_save_path = os.path.join(results_save_path, f"session_{session_idx + 1}_embeddings.png")
    plot_embeddings_side_by_side(
        embeddings_2d=embeddings_2d,
        embeddings_3d=embeddings_3d,
        umap_embeddings=umap_embeddings,
        session=session,
        hipp_angle_binned=hipp_angle_binned,
        true_angle_binned=true_angle_binned,
        principal_curve_3d=principal_curve_3d,
        save_path=plot_save_path
    )

    # Perform t-SNE embedding
    print("Performing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(neural_data)
    print(f"t-SNE embeddings shape: {tsne_embeddings.shape}")

    # Plot t-SNE embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=10, c=hipp_angle_binned, cmap='viridis')
    plt.title('t-SNE Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hippocampal Angle (degrees)')
    plt.show()
    print("Displayed t-SNE plot.")

    return

def calculate_single_H_neighbors(principal_curve=None, tt=None, embeddings=None, t0=None, t1=None, true_angle=None, n_neighbors=None):
    """
    Calculate the hippocampal gain (H) between two time points t0 and t1
    by averaging the tt values from the closest manifold coordinates of the n nearest neighbors
    of the embeddings at t0 and t1.
    
    Parameters:
    - principal_curve (np.ndarray): Array representing the principal manifold.
    - tt (np.ndarray): Array of parameterization along the principal curve.
    - embeddings (np.ndarray): Array of embedding vectors for each time point.
    - t0 (int): Starting time index.
    - t1 (int): Ending time index.
    - true_angle (np.ndarray): Array of true angles corresponding to each time point.
    - n_neighbors (int): Number of closest neighbors to consider (default=5).
    
    Returns:
    - H (float): Calculated hippocampal gain between t0 and t1.
    """
    
    # Ensure t0 and t1 are within the valid range
    if t0 >= len(embeddings) or t1 >= len(embeddings):
        raise IndexError(f"t0 or t1 is out of bounds for embeddings of length {len(embeddings)}.")
    
    # Fit Nearest Neighbors model on embeddings
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(embeddings)
    
    # Function to compute average tt value for a given embedding index
    def get_avg_tt(embedding_index):
        # Get the embedding at the specified index
        embedding_point = embeddings[embedding_index].reshape(1, -1)
        # Find the n_neighbors+1 closest embeddings (including itself)
        distances, indices = nbrs.kneighbors(embedding_point)
        # Exclude the embedding itself
        neighbor_indices = indices[0][1:]
        tt_values = []
        for idx in neighbor_indices:
            # Get the embedding of the neighbor
            embedding_neighbor = embeddings[idx].reshape(1, -1)
            # Find the closest manifold coordinate for the neighbor
            _, _, tt_index = fhf.get_closest_manifold_coords(principal_curve, tt, embedding_neighbor, return_all=True)
            tt_value = tt[tt_index]
            tt_values.append(tt_value)
        # Calculate the mean of the tt values
        # print(f"IN CALCULATE H, the tt values: {tt_values}")
        return np.mean(tt_values)
    
    # Get average tt values for t0 and t1
    avg_tt_t0 = get_avg_tt(t0)
    avg_tt_t1 = get_avg_tt(t1)
    
    # Compute the difference in true angles
    true_angle_diff = true_angle[t1] - true_angle[t0]
    if true_angle_diff == 0:
        print(f"Division by zero encountered for t0={t0} and t1={t1}. Adding a small epsilon to denominator.")
        true_angle_diff += 1e-9  # Add a small value to avoid division by zero

    # print(f"true_angle t0 : {true_angle[t0]}")
    # print(f"true_angle t1 : {true_angle[t1]}")
    # print(f"true_angle_diff is: {true_angle_diff}")
    # print(f"true_angle_diff mod is: {true_angle_diff % (2*np.pi)}")

    # print(f"avg_tt_t0 is: {avg_tt_t0}")
    # print(f"avg_tt_t1 is: {avg_tt_t1}")
    # print(f"avg_tt_diifis: {(avg_tt_t1 - avg_tt_t0)}")
    # print(f"avg_tt_diif mod is: {(avg_tt_t1 - avg_tt_t0) % (2*np.pi)}")


    # print(f"H value between {t0} and {t1}")
    
    # Calculate the hippocampal gain H
    H = (((avg_tt_t1 - avg_tt_t0) % (2*np.pi)) / ((true_angle_diff) % (2*np.pi)))

    return H

def calculate_single_H_array(decoded_angle_t0=None, decoded_angle_t1=None, true_angle_t0=None, true_angle_t1=None):

    print(f"Inputs:")
    print(f"  decoded_angle_t0: {decoded_angle_t0}")
    print(f"  decoded_angle_t1: {decoded_angle_t1}")
    print(f"  true_angle_t0: {true_angle_t0}")
    print(f"  true_angle_t1: {true_angle_t1}")
    
    true_diff = true_angle_t1 - true_angle_t0
    if true_diff == 0:
        true_diff = 1e-6
        print("  true_diff was zero, adjusted to 1e-6 to avoid division by zero.")
    print(f"  true_diff: {true_diff}")
    
    decoded_diff = (decoded_angle_t1 - decoded_angle_t0) % (2 * np.pi)
    print(f"  decoded_diff (wrapped to [0, 2π]): {decoded_diff}")
    
    wrapped_true_diff = true_diff % (2 * np.pi)
    print(f"  wrapped_true_diff (wrapped to [0, 2π]): {wrapped_true_diff}")
    
    H = decoded_diff / wrapped_true_diff
    print(f"  Calculated H: {H}")
    
    return H

def calculate_over_experiment_H_array(decoded_angles=None,true_angles=None,spacing=None):
    print(f"decoded: {len(decoded_angles)}")
    print(f"true: {len(true_angles)}")

    H_list = []
    for i in range(len(true_angles)-1-spacing):
        t0 = i
        t1 = i + spacing  
        H_temp = calculate_single_H_array(decoded_angle_t0=decoded_angles[t0], decoded_angle_t1=decoded_angles[t1], true_angle_t0=true_angles[t0], true_angle_t1=true_angles[t1])
        H_list.append(H_temp)


    return np.array(H_list)



def calculate_over_experiment_H(principal_curve=None, tt=None, embeddings=None,true_angle=None,num_avg_over=2, spacing=0, n_neighbors=5):

    """
    Calculate the hippocampal gain (H) over an entire experiment by averaging over specified time steps.
    
    Parameters:
    - principal_curve: Array representing the principal manifold.
    - tt: Array of temporal or spatial indices corresponding to the principal_curve.
    - embeddings: Array of embedding vectors for each time point.
    - true_angle: Array of true angles corresponding to each time point.
    - num_avg_over: Number of time steps to average over (default=3).
    - spacing: Spacing between time points to consider for averaging (default=2).
    
    Returns:
    - H_array: Array of averaged hippocampal gain values across the experiment.
    """
    
    H_list = []
    embeddings_length = len(embeddings)

    for i in range(((len(embeddings)-1)-spacing)):

        H_temp_list = []
        t0 = i
        for j in range(num_avg_over):
            t1 = i + j + spacing
            if t1 >= embeddings_length:
                raise IndexError(
                    f"Index t1={t1} out of bounds for embeddings of length {embeddings_length}."
                )
            H_temp = calculate_single_H_neighbors(principal_curve, tt, embeddings, t0, t1, true_angle,n_neighbors=n_neighbors)
            H_temp_list.append(H_temp)

        H_temp = np.mean(H_temp_list)
        H_list.append(H_temp)
    
    return np.array(H_list)

def plot_Hs(est_H=None, decode_H=None, behav_var=None, behav_var_name=None, session_idx=None, 
           session=None, save_path=None, tag=None, is_moving_avg=False, SI_score=None, decode_err=None):
    """
    Plots estimated hippocampal gain (est_H) against decoded gain (decode_H).
    Optionally, if moving averaged data and behavioral variables are provided, it plots them with color mapping.

    Parameters:
    - est_H (np.ndarray): Array of estimated gain values.
    - decode_H (np.ndarray): Array of decoded gain values.
    - behav_var (np.ndarray, optional): Array of behavioral variable values (e.g., velocity).
    - behav_var_name (str, optional): Name of the behavioral variable for labeling.
    - session_idx (int, optional): Session index for labeling purposes.
    - session (object, optional): Session object containing metadata. Expected to have attributes 'rat', 'day', and 'epoch'.
    - save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    - tag (str, optional): Additional tag for the plot title and filename.
    - is_moving_avg (bool, optional): Indicates if the data provided are moving averaged.
    - SI_score (float, optional): SI score to display on the plot.
    - decode_err (float, optional): Decode error to display on the plot.

    Returns:
    - None
    """
    
    # Validate input arrays
    if est_H is None or decode_H is None:
        raise ValueError("est_H and decode_H must both be provided.")
    
    if behav_var is not None and behav_var_name is None:
        raise ValueError("behav_var_name must be provided if behav_var is used.")
    
    # Find the overlapping range of indices
    min_length = min(len(est_H), len(decode_H))
    if behav_var is not None:
        min_length = min(min_length, len(behav_var))
    
    # Trim est_H, decode_H, and behav_var to the overlapping range
    est_gain_trimmed = est_H[:min_length]
    decode_gain_trimmed = decode_H[:min_length]
    times = np.arange(min_length)  # Time is just the index
    
    if behav_var is not None:
        behav_var_trimmed = behav_var[:min_length]
    
    # Handle moving average adjustments if needed
    if is_moving_avg and behav_var is not None:
        # Assuming moving average was computed with 'valid' mode
        # The trimmed arrays are already moving averaged, so times should be adjusted
        # This requires the window_size to have been considered during averaging
        # For simplicity, we'll assume the user has adjusted the times accordingly
        pass  # No action needed; user is responsible for providing correct times
    
    # Calculate the overall averages of the data
    avg_est_gain = np.mean(est_gain_trimmed)
    avg_decode_gain = np.mean(decode_gain_trimmed)
    if behav_var is not None:
        avg_behav_var = np.mean(behav_var_trimmed)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if behav_var is not None:
        # Normalize the behavioral variable for color mapping
        norm = mcolors.Normalize(vmin=np.min(behav_var_trimmed), vmax=np.max(behav_var_trimmed))
        cmap = plt.get_cmap('viridis')  # Choose a suitable colormap
        
        # Create a ScalarMappable for the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for older versions of matplotlib
        
        # Create LineCollection for est_H
        points_est = np.array([times, est_gain_trimmed]).T.reshape(-1, 1, 2)
        segments_est = np.concatenate([points_est[:-1], points_est[1:]], axis=1)
        lc_est = LineCollection(segments_est, colors=cmap(norm(behav_var_trimmed[:-1])), linewidths=2, label='Estimated Gain')
        
        # Create LineCollection for decode_H
        points_decode = np.array([times, decode_gain_trimmed]).T.reshape(-1, 1, 2)
        segments_decode = np.concatenate([points_decode[:-1], points_decode[1:]], axis=1)
        lc_decode = LineCollection(segments_decode, colors=cmap(norm(behav_var_trimmed[:-1])), linewidths=2, label='Decoded Gain', alpha=0.7)
        
        # Add LineCollections to the Axes
        ax.add_collection(lc_est)
        ax.add_collection(lc_decode)
        
        # Add the colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f'{behav_var_name}', fontsize=12)
    else:
        # Plot without behavioral variable coloring
        ax.plot(times, est_gain_trimmed, label='Estimated Gain', color='blue')
        ax.plot(times, decode_gain_trimmed, label='Decoded Gain', color='red', alpha=0.7)
    
    # Add dummy plot lines for the legend if using LineCollection
    if behav_var is not None:
        ax.plot([], [], color='blue', label='Estimated Gain')
        ax.plot([], [], color='red', alpha=0.7, label='Decoded Gain')
    
    # Prepare the text for average values
    avg_text = (
        f'Overall Avg Estimated Gain: {avg_est_gain:.2f}\n'
        f'Overall Avg Decoded Gain: {avg_decode_gain:.2f}\n'
    )
    if behav_var is not None:
        avg_text += f'Overall Avg Behavioral Var: {avg_behav_var:.2f}\n'
    if SI_score is not None:
        avg_text += f'SI Score: {SI_score:.2f}\n'
    if decode_err is not None:
        avg_text += f'Avg Decode Err: {decode_err:.2f}'
    
    # Add text annotation in the top-left corner
    ax.text(
        0.05, 0.95, avg_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5)
    )
    
    # Setting labels and title
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Gain', fontsize=14)
    
    title = 'FT Gain vs Spline Decoded Gain'
    if is_moving_avg:
        title += ' (Moving Averages)'
    if behav_var is not None:
        title += f' Colored by {behav_var_name}'
    
    # Adding session information to the title if available
    if session_idx is not None and session is not None:
        # Ensure that session has attributes 'rat', 'day', and 'epoch'
        rat = getattr(session, 'rat', 'Unknown Rat')
        day = getattr(session, 'day', 'Unknown Day')
        epoch = getattr(session, 'epoch', 'Unknown Epoch')
        title += f'\nSession {session_idx}: Rat {rat}, Day {day}, Epoch {epoch}'
    
    # Include the tag in the title if provided
    if tag is not None:
        title += f', Tag: {tag}'
    
    ax.set_title(title, fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    
    # Set the limits based on the data
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(
        min(np.min(est_gain_trimmed), np.min(decode_gain_trimmed)) - 0.1, 
        max(np.max(est_gain_trimmed), np.max(decode_gain_trimmed)) + 0.1
    )
    
    # Save or show the plot
    if save_path:
        if session_idx is not None:
            # Define the directory structure
            save_dir = os.path.join(save_path, 'h_plots', f'session_{session_idx}')
        else:
            save_dir = os.path.join(save_path, 'h_plots')
        os.makedirs(save_dir, exist_ok=True)
        
        # Define the filename with tag if provided
        filename = f"h_est_vs_decode"
        if is_moving_avg:
            filename += "_ma"
        if tag:
            filename += f"_{tag}"
        filename += ".png"
        
        full_save_path = os.path.join(save_dir, filename)
        
        # Save the figure
        fig.savefig(full_save_path, dpi=300)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved plot to {full_save_path}")
    else:
        plt.show()

def plot_Hs_over_laps(est_H=None, decode_H=None, lap_number=None, session_idx=None, session=None, 
                      save_path=None, tag=None, SI_score=None, decode_err=None):
    """
    Plots two H values against lap numbers, optionally annotating with session information and saving the plot.
    
    Parameters:
    - est_H (np.ndarray): Array of the first H values to plot.
    - decode_H (np.ndarray): Array of the second H values to plot.
    - lap_number (np.ndarray): Array of lap numbers corresponding to each H value.
    - session_idx (int, optional): Session index for labeling purposes.
    - session (object, optional): Session object containing metadata (e.g., rat ID, day, epoch).
    - save_path (str, optional): Directory path to save the plot. If None, the plot is displayed.
    - tag (str, optional): Tag to include in the saved plot's filename.
    - SI_score (float, optional): Score to annotate on the plot.
    - decode_err (float, optional): Decoding error to annotate on the plot.
    
    Returns:
    - None
    """

    print(f"lap number first 20: {lap_number[:20]}")
    
    # Input validation
    if est_H is None or decode_H is None or lap_number is None:
        raise ValueError("est_H, decode_H, and lap_number must all be provided.")
    
    est_H = np.asarray(est_H)
    decode_H = np.asarray(decode_H)
    lap_number = np.asarray(lap_number)
    
    if est_H.ndim != 1 or decode_H.ndim != 1 or lap_number.ndim != 1:
        raise ValueError("est_H, decode_H, and lap_number must all be 1-dimensional arrays.")
    
    
    # Find the overlapping range of indices
    min_length = min(len(est_H), len(decode_H), len(lap_number))
    
    # Trim est_H, decode_H, and lap_number to the overlapping range
    est_H_trimmed = est_H[:min_length]
    decode_H_trimmed = decode_H[:min_length]
    lap_trimmed = lap_number[:min_length]
    
    # Compute average values for annotations
    avg_est_H = np.mean(est_H_trimmed)
    avg_decode_H = np.mean(decode_H_trimmed)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(lap_trimmed, est_H_trimmed, label='est_H Value', color='blue', marker='o', linestyle='-')
    plt.plot(lap_trimmed, decode_H_trimmed, label='decode_H Value', color='red', marker='x', linestyle='--')
    
    # Create an axes instance
    ax = plt.gca()
    
    # Prepare annotation text
    annotation_text = f'Avg est_H: {avg_est_H:.2f}\nAvg decode_H: {avg_decode_H:.2f}'
    if SI_score is not None:
        annotation_text += f'\nSI Score: {SI_score:.2f}'
    if decode_err is not None:
        annotation_text += f'\nDecode Error: {decode_err:.2f}'
    
    # Add text annotation in the top-left corner
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    plt.xlabel('Lap Number', fontsize=14)
    plt.ylabel('H Value', fontsize=14)
    title = 'est_H and decode_H Values Over Laps'
    
    if session_idx is not None and session is not None:
        # Assuming session object has attributes: rat, day, epoch
        title += f'\nSession {session_idx}: Rat {session.rat}, Day {session.day}, Epoch {session.epoch}'
        if tag is not None:
            title += f', Tag {tag}'
    elif tag is not None:
        title += f'\nTag: {tag}'
    
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        # Construct the directory path
        dir_components = [save_path, 'h_plots']
        if session_idx is not None:
            dir_components.append(f'session_{session_idx}')
        dir_path = os.path.join(*dir_components)
        os.makedirs(dir_path, exist_ok=True)
        
        # Construct the filename
        filename = f"h_over_laps_{tag}.png" if tag else "h_over_laps.png"
        full_path = os.path.join(dir_path, filename)
        
        plt.savefig(full_path, dpi=300)
        plt.close()
        
        print(f"Saved plot to {full_path}")
    else:
        plt.show()

def plot_Hs_over_laps_moving_average(est_H=None, decode_H=None, lap_number=None, window_size=20, 
                                    session_idx=None, session=None, 
                                    save_path=None, tag=None, SI_score=None, decode_err=None):
    """
    Plots two H values (est_H and decode_H) against lap numbers with a moving average, 
    optionally annotating with session information and saving the plot.
    
    Parameters:
    - est_H (np.ndarray): Array of the first H values to plot.
    - decode_H (np.ndarray): Array of the second H values to plot.
    - lap_number (np.ndarray): Array of lap numbers corresponding to each H value.
    - window_size (int, optional): Size of the moving average window. Defaults to 100.
    - session_idx (int, optional): Session index for labeling purposes.
    - session (object, optional): Session object containing metadata (e.g., rat ID, day, epoch).
    - save_path (str, optional): Directory path to save the plot. If None, the plot is displayed.
    - tag (str, optional): Tag to include in the saved plot's filename.
    - SI_score (float, optional): Score to annotate on the plot.
    - decode_err (float, optional): Decoding error to annotate on the plot.
    
    Returns:
    - None
    """
    
    # Input validation
    if est_H is None or decode_H is None or lap_number is None:
        raise ValueError("est_H, decode_H, and lap_number must all be provided.")
    
    est_H = np.asarray(est_H)
    decode_H = np.asarray(decode_H)
    lap_number = np.asarray(lap_number)
    
    if est_H.ndim != 1 or decode_H.ndim != 1 or lap_number.ndim != 1:
        raise ValueError("est_H, decode_H, and lap_number must all be 1-dimensional arrays.")
    
    # Determine the minimum length among the inputs
    min_length = min(len(est_H), len(decode_H), len(lap_number))
    
    # Check if trimming is necessary
    if not (len(est_H) == len(decode_H) == len(lap_number)):
        print(f"Input arrays have different lengths. Trimming to the smallest length: {min_length}.")
    
    # Trim est_H, decode_H, and lap_number to the overlapping range
    est_H_trimmed = est_H[:min_length]
    decode_H_trimmed = decode_H[:min_length]
    lap_trimmed = lap_number[:min_length]
    
    # Validate window_size
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer.")
    if window_size > min_length:
        raise ValueError(f"window_size ({window_size}) is larger than the data length ({min_length}).")
    
    # Function to compute moving average
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    # Apply moving average
    est_H_ma = moving_average(est_H_trimmed, window_size)
    decode_H_ma = moving_average(decode_H_trimmed, window_size)
    
    # Compute average values for annotations based on moving average
    avg_est_H = np.mean(est_H_ma)
    avg_decode_H = np.mean(decode_H_ma)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(lap_trimmed, est_H_ma, label='est_H (Moving Avg)', color='blue', linestyle='-')
    plt.plot(lap_trimmed, decode_H_ma, label='decode_H (Moving Avg)', color='red', linestyle='--')
    
    # Create an axes instance
    ax = plt.gca()
    
    # Prepare annotation text
    annotation_text = f'Avg est_H (MA): {avg_est_H:.2f}\nAvg decode_H (MA): {avg_decode_H:.2f}'
    if SI_score is not None:
        annotation_text += f'\nSI Score: {SI_score:.2f}'
    if decode_err is not None:
        annotation_text += f'\nDecode Error: {decode_err:.2f}'
    
    # Add text annotation in the top-left corner
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    plt.xlabel('Lap Number', fontsize=14)
    plt.ylabel('H Value (Moving Average)', fontsize=14)
    title = 'est_H and decode_H Values Over Laps (Moving Average)'
    
    if session_idx is not None and session is not None:
        # Assuming session object has attributes: rat, day, epoch
        title += f'\nSession {session_idx}: Rat {session.rat}, Day {session.day}, Epoch {session.epoch}'
        if tag is not None:
            title += f', Tag {tag}'
    elif tag is not None:
        title += f'\nTag: {tag}'
    
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        # Construct the directory path
        dir_components = [save_path, 'h_plots']
        if session_idx is not None:
            dir_components.append(f'session_{session_idx}')
        dir_path = os.path.join(*dir_components)
        os.makedirs(dir_path, exist_ok=True)
        
        # Construct the filename
        if tag:
            filename = f"h_over_laps_MA_{tag}.png"
        else:
            filename = "h_over_laps_MA.png"
        full_path = os.path.join(dir_path, filename)
        
        plt.savefig(full_path, dpi=300)
        plt.close()
        
        print(f"Saved plot to {full_path}")
    else:
        plt.show()

def plot_Hs_over_laps_interactive(est_H=None, decode_H=None, lap_number=None, session_idx=None, session=None, 
                                  save_path=None, tag=None, SI_score=None, decode_err=None,mean_diff=None,std_diff=None):
    """
    Plots two H values against lap numbers using Plotly for interactivity.

    Parameters:
    - est_H (np.ndarray): Array of the first H values to plot.
    - decode_H (np.ndarray): Array of the second H values to plot.
    - lap_number (np.ndarray): Array of lap numbers corresponding to each H value.
    - session_idx (int, optional): Session index for labeling purposes.
    - session (object, optional): Session object containing metadata.
    - save_path (str, optional): Directory path to save the plot HTML file.
    - tag (str, optional): Tag to include in the saved plot's filename.
    - SI_score (float, optional): Score to annotate on the plot.
    - decode_err (float, optional): Decoding error to annotate on the plot.

    Returns:
    - None
    """
    
    # Input validation
    if est_H is None or decode_H is None or lap_number is None:
        raise ValueError("est_H, decode_H, and lap_number must all be provided.")
    
    est_H = np.asarray(est_H)
    decode_H = np.asarray(decode_H)
    lap_number = np.asarray(lap_number)
    
    if est_H.ndim != 1 or decode_H.ndim != 1 or lap_number.ndim != 1:
        raise ValueError("est_H, decode_H, and lap_number must all be 1-dimensional arrays.")
    
    # Find the overlapping range of indices
    min_length = min(len(est_H), len(decode_H), len(lap_number))
    
    # Trim est_H, decode_H, and lap_number to the overlapping range
    est_H_trimmed = est_H[:min_length]
    decode_H_trimmed = decode_H[:min_length]
    lap_trimmed = lap_number[:min_length]
    
    # Compute average values for annotations
    avg_est_H = np.mean(est_H_trimmed)
    avg_decode_H = np.mean(decode_H_trimmed)

    # Create interactive plot
    trace_est_H = go.Scatter(
        x=lap_trimmed,
        y=est_H_trimmed,
        mode='lines+markers',
        name='est_H Value',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
    )

    trace_decode_H = go.Scatter(
        x=lap_trimmed,
        y=decode_H_trimmed,
        mode='lines+markers',
        name='decode_H Value',
        line=dict(color='red', dash='dash'),
        marker=dict(symbol='x')
    )

    data = [trace_est_H, trace_decode_H]

    # Prepare annotation text
    annotation_text = f'Avg est_H: {avg_est_H:.2f}<br>Avg decode_H: {avg_decode_H:.2f}'
    if SI_score is not None:
        annotation_text += f'<br>SI Score: {SI_score:.2f}'
    if decode_err is not None:
        annotation_text += f'<br>Decode Error: {decode_err:.2f}'
    if mean_diff is not None:
        annotation_text += f'<br>mean_diff: {mean_diff:.2f}'
    if std_diff is not None:
        annotation_text += f'<br>std_diff: {std_diff:.2f}'

    # Define the base title text with session information if available
    base_title = 'est_H and decode_H Values Over Laps'
    if session_idx is not None and session is not None:
        base_title += f'<br>Session {session_idx}: Rat {session.rat}, Day {session.day}, Epoch {session.epoch}'
        if tag is not None:
            base_title += f', Tag {tag}'
    elif tag is not None:
        base_title += f'<br>Tag: {tag}'

    # Create layout with title as a dictionary and annotations
    layout = go.Layout(
        title={'text': base_title},  # Set the title text correctly as a dictionary
        xaxis=dict(title='Lap Number'),
        yaxis=dict(title='H Value'),
        annotations=[
            dict(
                xref='paper', yref='paper',
                x=0.01, y=0.99,
                xanchor='left', yanchor='top',
                text=annotation_text,
                showarrow=False,
                font=dict(size=12),
                bordercolor='black',
                borderwidth=1,
                borderpad=5,
                bgcolor='white',
                opacity=0.8
            )
        ]
    )

    fig = go.Figure(data=data, layout=layout)

    # Save or show the plot
    if save_path:
        # Construct the directory path
        dir_components = [save_path, 'h_plots_interactive']
        if session_idx is not None:
            dir_components.append(f'session_{session_idx}')
        dir_path = os.path.join(*dir_components)
        os.makedirs(dir_path, exist_ok=True)

        # Construct the filename
        filename = f"h_over_laps_{tag}.html" if tag else "h_over_laps.html"
        full_path = os.path.join(dir_path, filename)

        # Save the plot as an HTML file
        plot(fig, filename=full_path, auto_open=False)
        print(f"Saved interactive plot to {full_path}")
    else:
        # Display the plot in the default web browser
        plot(fig)

def plot_decoded_var_and_true_interactive(decoded_var=None, behav_var=None, true_angle=None,
                                         indices=None,
                                         xlabel='Time (seconds)', 
                                         ylabel1='Decoded Variable', 
                                         ylabel2='Behavioral Variable', 
                                         legend_labels=['Decoded Variable', 'Behavioral Variable', 'True Angle'],
                                         save_path=None,
                                         session_idx=None,
                                         behav_var_name=None):
    """
    Plots decoded_var, behav_var, and true_angle interactively using Plotly and saves the plot as an HTML file.

    Parameters:
    - decoded_var (array-like): Array of decoded variables to plot.
    - behav_var (array-like): Array of behavioral variables to plot.
    - true_angle (array-like, optional): Array of true angle variables to plot.
    - indices (list or array-like, optional): Specific indices to plot. Defaults to plotting the entire array.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel1 (str, optional): Label for the decoded_var.
    - ylabel2 (str, optional): Label for the behav_var.
    - title (str, optional): Plot title.
    - legend_labels (list, optional): Labels for the three variables.
    - save_path (str, optional): Directory path to save the plot.
    - session_idx (int, optional): Session index for labeling purposes.
    - behav_var_name (str, optional): Name of the behavioral variable for labeling and filename.

    Returns:
    - None
    """
    min_length = min(len(decoded_var), len(behav_var), len(true_angle))
    decoded_angles = decoded_var[:min_length]
    binned_hipp_angle_rad = behav_var[:min_length]
    binned_true_angle_rad = true_angle[:min_length]
    # Set default indices to plot the entire array if not provided
    if indices is None:
        indices = np.arange(len(decoded_var))
    else:
        if isinstance(indices, slice):
            indices = np.arange(len(decoded_var))[indices]
        elif isinstance(indices, int):
            indices = np.array([indices])
        else:
            indices = np.array(indices)
    
    # Extract the data for the specified indices
    decoded_subset = np.array(decoded_var)[indices]
    behav_subset = np.array(behav_var)[indices]
    x_values = indices  # Assuming each index represents one second
    
    if true_angle is not None:
        true_angle_subset = np.array(true_angle)[indices]
    
    # Create Plotly traces
    trace_decoded = go.Scatter(
        x=x_values,
        y=decoded_subset,
        mode='lines+markers',
        name=legend_labels[0],
        marker=dict(color='blue', symbol='circle'),
        line=dict(color='blue')
    )
    
    trace_behav = go.Scatter(
        x=x_values,
        y=behav_subset,
        mode='lines+markers',
        name=legend_labels[1],
        marker=dict(color='red', symbol='x'),
        line=dict(color='red', dash='dash')
    )
    
    data = [trace_decoded, trace_behav]
    
    if true_angle is not None:
        trace_true_angle = go.Scatter(
            x=x_values,
            y=true_angle_subset,
            mode='lines+markers',
            name=legend_labels[2],
            marker=dict(color='green', symbol='square'),
            line=dict(color='green', dash='dot')
        )
        data.append(trace_true_angle)
    
    # Define the layout
    layout = go.Layout(
        title=behav_var_name if session_idx is None else f"{behav_var_name} - Session {session_idx}",
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel1),
        legend=dict(x=0, y=1.0),
        hovermode='closest'
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    # Save the plot if save_path is provided
    if save_path:
        full_save_path = os.path.join(save_path, f'session_{session_idx}' if session_idx is not None else '')
        os.makedirs(full_save_path, exist_ok=True)
        
        filename = f"SI_{behav_var_name}.html" if behav_var_name else "decoded_and_behavioral_variables.html"
        file_path = os.path.join(full_save_path, filename)
        
        plot(fig, filename=file_path, auto_open=False)
        print(f"Interactive plot saved to {file_path}")
    
    # Display the plot
    fig.show()




def plot_Hs_moving_avg(est_H=None, decode_H=None, behav_var=None,behav_var_name=None, session_idx=None, 
                             session=None, save_path=None, tag=None, window_size=5,SI_score=None,decode_err=None):
    """
    Plots the moving averages of estimated hippocampal gain (est_H) against decoded gain (decode_H),
    with colors representing a behavioral variable (e.g., velocity).
    
    Parameters:
    - est_H (np.ndarray): Array of estimated gain values.
    - decode_H (np.ndarray): Array of decoded gain values.
    - behav_var (np.ndarray): Array of behavioral variable values (e.g., velocity).
    - session_idx (int, optional): Session index for labeling purposes.
    - session (object, optional): Session object containing metadata. Expected to have attributes 'rat', 'day', and 'epoch'.
    - save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    - tag (str, optional): Additional tag for the plot title and filename.
    - window_size (int, optional): Number of time steps to average over for smoothing (default=5).
    
    Returns:
    - None
    """
    
    # Validate input arrays
    if est_H is None or decode_H is None or behav_var is None:
        raise ValueError("est_H, decode_H, and behav_var must all be provided.")
    
    # Validate window_size
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    
    # Find the overlapping range of indices
    min_length = min(len(est_H), len(decode_H), len(behav_var))
    
    # Trim est_H, decode_H, and behav_var to the overlapping range
    est_gain_trimmed = est_H[:min_length]
    decode_gain_trimmed = decode_H[:min_length]
    behav_var_trimmed = behav_var[:min_length]
    times = np.arange(min_length)  # Time is just the index
    
    # Optional: Handle NaNs and Infs if necessary
    # Uncomment and adjust the following lines if your data contains NaNs or Infs
    # valid_indices = ~np.isnan(est_gain_trimmed) & ~np.isnan(decode_gain_trimmed) & \
    #                 ~np.isnan(behav_var_trimmed) & ~np.isinf(est_gain_trimmed) & \
    #                 ~np.isinf(decode_gain_trimmed) & ~np.isinf(behav_var_trimmed)
    # est_gain_trimmed = est_gain_trimmed[valid_indices]
    # decode_gain_trimmed = decode_gain_trimmed[valid_indices]
    # behav_var_trimmed = behav_var_trimmed[valid_indices]
    # times = times[valid_indices]
    
    # Check if the trimmed arrays are long enough for the moving average
    if min_length < window_size:
        raise ValueError(f"Not enough data points ({min_length}) for the specified window_size ({window_size}).")
    
    # Compute moving averages using numpy's convolution
    kernel = np.ones(window_size) / window_size
    avg_est_gain = np.convolve(est_gain_trimmed, kernel, mode='valid')
    avg_decode_gain = np.convolve(decode_gain_trimmed, kernel, mode='valid')
    avg_behav_var = np.convolve(behav_var_trimmed, kernel, mode='valid')
    
    # Adjust the time axis for 'valid' mode convolution
    # This centers the window; alternatively, you can adjust as needed
    adjusted_times = times[(window_size - 1)//2 : -(window_size//2)] if window_size > 1 else times
    
    # Calculate the overall averages of the moving averages
    overall_avg_est_gain = np.mean(avg_est_gain)
    overall_avg_decode_gain = np.mean(avg_decode_gain)
    overall_avg_behav_var = np.mean(avg_behav_var)
    
    # Normalize the behavioral variable for color mapping
    norm = mcolors.Normalize(vmin=np.min(avg_behav_var), vmax=np.max(avg_behav_var))
    cmap = plt.get_cmap('viridis')  # Choose a suitable colormap
    
    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    
    # Create the plot using object-oriented Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create LineCollection for est_H
    points_est = np.array([adjusted_times, avg_est_gain]).T.reshape(-1, 1, 2)
    segments_est = np.concatenate([points_est[:-1], points_est[1:]], axis=1)
    lc_est = LineCollection(segments_est, colors=cmap(norm(avg_behav_var[:-1])), linewidths=2, label=f'Estimated Gain (MA window={window_size})')
    
    # Create LineCollection for decode_H
    points_decode = np.array([adjusted_times, avg_decode_gain]).T.reshape(-1, 1, 2)
    segments_decode = np.concatenate([points_decode[:-1], points_decode[1:]], axis=1)
    lc_decode = LineCollection(segments_decode, colors=cmap(norm(avg_behav_var[:-1])), linewidths=2, label=f'Decoded Gain (MA window={window_size})', alpha=0.7)
    
    # Add LineCollections to the Axes
    ax.add_collection(lc_est)
    ax.add_collection(lc_decode)
    
    # Add dummy plot lines for the legend
    ax.plot([], [], color='blue', label=f'Estimated Gain (MA window={window_size})')
    ax.plot([], [], color='red', alpha=0.7, label=f'Decoded Gain (MA window={window_size})')
    
    # Add the colorbar to the figure
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f'{behav_var_name}', fontsize=12)
    
    # Prepare the text for average values
    avg_text = (
        f'Overall Avg Estimated Gain: {overall_avg_est_gain:.2f}\n'
        f'Overall Avg Decoded Gain: {overall_avg_decode_gain:.2f}\n'
        f'Overall Avg Behavioral Var: {overall_avg_behav_var:.2f}\n'
        f'SI Score: {SI_score:.2f}\n'
        f'Avg_decode_err: {decode_err:.2f}'
    )
    
    # Add text annotation in the top-left corner
    ax.text(
        0.05, 0.95, avg_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5)
    )
    
    # Setting labels and title
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Gain', fontsize=14)
    title = 'FT Gain vs Spline Decoded Gain (Moving Averages) Colored by Behavioral Variable'
    
    # Adding session information to the title if available
    if session_idx is not None and session is not None:
        # Ensure that session has attributes 'rat', 'day', and 'epoch'
        rat = getattr(session, 'rat', 'Unknown Rat')
        day = getattr(session, 'day', 'Unknown Day')
        epoch = getattr(session, 'epoch', 'Unknown Epoch')
        title += f'\nSession {session_idx}: Rat {rat}, Day {day}, Epoch {epoch}'
    
    # Include the tag in the title if provided
    if tag is not None:
        title += f', Tag: {tag}'
    
    ax.set_title(title, fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    
    # Set the limits based on the data
    ax.set_xlim(adjusted_times.min(), adjusted_times.max())
    ax.set_ylim(
        min(np.min(avg_est_gain), np.min(avg_decode_gain)) - 0.1, 
        max(np.max(avg_est_gain), np.max(avg_decode_gain)) + 0.1
    )
    
    # Save or show the plot
    if save_path:
        # Define the directory structure
        save_dir = os.path.join(save_path, 'h_plots', f'session_{session_idx}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Define the filename with tag if provided
        filename = f"h_est_vs_decode_ma_window{window_size}_{tag}.png" if tag else f"h_est_vs_decode_ma_window{window_size}.png"
        full_save_path = os.path.join(save_dir, filename)
        
        # Save the figure
        fig.savefig(full_save_path, dpi=300)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved plot to {full_save_path}")
    else:
        plt.show()

def compute_SI_and_plot(embeddings=None,behav_var=None,params=None,behav_var_name=None,save_dir=None,session_idx=None):

    """ 
    params: 

    n_bins: integer (default: 10)
        number of bin-groups the label will be divided into (they will 
        become nodes on the graph). For vectorial features, if one wants 
        different number of bins for each entry then specify n_bins as a 
        list (i.e. [10,20,5]). Note that it will be ignored if 
        'discrete_label' is set to True.

    n_neighbors: int (default: 15)
        Number of neighbors used to compute the overlapping between 
        bin-groups. This parameter controls the tradeoff between local and 
        global structure.

    discrete_label: boolean (default: False)
        If the label is discrete, then one bin-group will be created for 
        each discrete value it takes. Note that if set to True, 'n_bins' 
        parameter will be ignored.

    num_shuffles: int (default: 100)
        Number of shuffles to be computed. Note it must fall within the 
        interval [0, np.inf).

    verbose: boolean (default: False)
        Boolean controling whether or not to print internal process..

    """

    SI, binLabel, overlap_mat, sSI = compute_structure_index(embeddings,behav_var,**params)

    print(f"SI  is: {SI}")

    #draw_graph(overlap_mat, ax)

    # Create the figure and subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 3D scatter
    at = fig.add_subplot(1, 3, 1, projection='3d')
    scatter = at.scatter(*embeddings.T, c=behav_var, cmap='inferno_r', vmin=0, vmax=1)
    cbar = fig.colorbar(scatter, ax=at, anchor=(0, 0.3), shrink=0.8, ticks=[0, 0.5, 1])
    cbar.set_label('radius', rotation=90)
    at.set_title(f'Embeddings with {behav_var_name} feature', size=16)
    at.set_xlabel('Dim 1', labelpad=-8, size=14)
    at.set_ylabel('Dim 2', labelpad=-8, size=14)
    at.set_zlabel('Dim 3', labelpad=-8, size=14)
    at.set_xticks([])
    at.set_yticks([])
    at.set_zticks([])

    # Plot adjacency matrix
    matshow = ax[1].matshow(overlap_mat, vmin=0, vmax=0.5, cmap=plt.cm.viridis)
    ax[1].xaxis.set_ticks_position('bottom')
    cbar = fig.colorbar(matshow, ax=ax[1], anchor=(0, 0.2), shrink=1, ticks=[0, 0.25, 0.5])
    cbar.set_label('overlap score', rotation=90, fontsize=14)
    ax[1].set_title('Adjacency matrix', size=16)
    ax[1].set_xlabel('bin-groups', size=14)
    ax[1].set_ylabel('bin-groups', size=14)

    # Plot weighted directed graph
    draw_graph(overlap_mat, ax[2], node_cmap=plt.cm.inferno_r, edge_cmap=plt.cm.Greys,
            node_names=np.round(binLabel[1][:, 0, 1], 2))
    ax[2].set_xlim(1.2 * np.array(ax[2].get_xlim()))
    ax[2].set_ylim(1.2 * np.array(ax[2].get_ylim()))
    ax[2].set_title('Directed graph', size=16)
    ax[2].text(0.98, 0.05, f"SI: {SI:.2f}", horizontalalignment='right',
            verticalalignment='bottom', transform=ax[2].transAxes, fontsize=25)
    
    filename = f"SI_{behav_var_name}.png"
    full_save_path = os.path.join(save_dir, f'session_{session_idx}')
    os.makedirs(full_save_path,exist_ok=True)

    # Adjust layout and show the plot
    plt.tight_layout()

    fig.savefig(os.path.join(full_save_path,filename), format='png', dpi=300, bbox_inches='tight')      

    plt.show()

    return SI

# def plot_decoded_var_and_true(tt, behav_var, 
#                          x_start=0, x_end=0.05, 
#                          xlabel='Parametrization (0 to 1)', 
#                          ylabel1='tt', ylabel2='Behavioral Variable', 
#                          title='tt and Behavioral Variable Plot', 
#                          legend_labels=['tt', 'Behavioral Variable'],
#                          save_path=None,
#                          figsize=(12, 6),session_idx=None,behav_var_name=None):
#     """
#     Plots two arrays of different sizes on the same x-axis ranging from x_start to x_end.
    
#     Parameters:
#     - tt (array-like): Array representing parametrization (e.g., time or parameter values).
#     - behav_var (array-like): Array representing the behavioral variable of interest.
#     - x_start (float): Starting value of the x-axis range. Default is 0.
#     - x_end (float): Ending value of the x-axis range. Default is 2π.
#     - xlabel (str): Label for the x-axis.
#     - ylabel1 (str): Label for the first y-axis (tt).
#     - ylabel2 (str): Label for the second y-axis (Behavioral Variable).
#     - title (str): Title of the plot.
#     - legend_labels (list of str): Labels for the legend entries.
#     - save_path (str): If provided, the directory path where the plot will be saved.
#     - figsize (tuple): Size of the figure in inches. Default is (12, 6).
    
#     Returns:
#     - None
#     """

    
#     # Convert inputs to numpy arrays for consistency
#     decoded_var = np.array(decoded_var)
#     behav_var = np.array(behav_var)
    
#     # Find the indices corresponding to the x_end value
#     # Assuming `decoded_var` is scaled between 0 and 1 or already aligned with your x-axis
#     end_index = np.searchsorted(decoded_var, x_end)
    
#     # Slice the data up to the end_index
#     decoded_var = decoded_var[:end_index]
#     behav_var = behav_var[:end_index]
    
#     # Normalize the x-axis indices for both arrays
#     decoded_var_length = len(decoded_var)
#     behav_length = len(behav_var)
#     x_decoded_var = np.linspace(x_start, x_end, decoded_var_length)
#     x_behav = np.linspace(x_start, x_end, behav_length)
    
#     # Create the plot
#     plt.figure(figsize=figsize)
    
#     # Plot decoded_var
#     plt.plot(x_decoded_var, decoded_var, label=legend_labels[0], linewidth=2)
    
#     # Plot behavioral variable
#     plt.plot(x_behav, behav_var % (2*np.pi), label=legend_labels[1], linewidth=2)
    
#     # Customize the plot
#     plt.xlabel(xlabel, fontsize=14)
#     plt.ylabel('Value', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.5)
    
#     # Set x-axis limits
#     plt.xlim(x_start, x_end)
    
#     # Optionally save the plot
#     if save_path:
#         # Ensure the save directory exists
#         filename = f"decoded_var_{behav_var_name}.png"
#         full_save_path = os.path.join(save_path, f'session_{session_idx}')
#         os.makedirs(full_save_path,exist_ok=True)

#         # Adjust layout and show the plot
#         plt.tight_layout()

#         plt.savefig(os.path.join(full_save_path,filename), format='png', dpi=300, bbox_inches='tight')      

#         print(f"Plot saved to {save_path}")
    
#     # Display the plot
#     plt.show()

def plot_decoded_var_and_true(decoded_var, behav_var, 
                              indices=None,
                              xlabel='Parametrization (0 to 1)', 
                              ylabel1='Decoded Variable', 
                              ylabel2='Behavioral Variable', 
                              title='Decoded and Behavioral Variable Plot', 
                              legend_labels=['Decoded Variable', 'Behavioral Variable'],
                              save_path=None,
                              figsize=(12, 6),
                              session_idx=None,
                              behav_var_name=None):
    """
    Plots decoded_var and behav_var over specified indices.

    Parameters:
    - decoded_var: array-like, decoded variable to plot.
    - behav_var: array-like, behavioral variable to plot.
    - indices: list or array-like, indices to plot. Defaults to first 20 elements.
    - xlabel: str, label for x-axis.
    - ylabel1: str, label for decoded_var.
    - ylabel2: str, label for behav_var.
    - title: str, plot title.
    - legend_labels: list of two strings, labels for the two variables.
    - save_path: str or None, path to save the plot.
    - figsize: tuple, figure size.
    - session_idx: optional, for additional information in title.
    - behav_var_name: optional, for labeling the behavioral variable.

    Returns:
    - None
    """
    # Set default number of points if indices not provided
    n = 250
    if indices is None:
        indices = range(n)
    else:
        # Validate indices
        if isinstance(indices, slice):
            # Convert slice to range
            start, stop, step = indices.start or 0, indices.stop or len(decoded_var), indices.step or 1
            indices = range(start, stop, step)
        elif not hasattr(indices, '__iter__'):
            raise TypeError("indices must be a slice or an iterable of integers")
        else:
            indices = list(indices)
    
    # Ensure decoded_var and behav_var have the same length
    if len(decoded_var) != len(behav_var):
        raise ValueError("decoded_var and behav_var must have the same length")
    
    # Extract the data for the specified indices
    decoded_subset = [decoded_var[i] for i in indices]
    behav_subset = [behav_var[i] for i in indices]
    x_values = list(indices)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(x_values, decoded_subset, label=legend_labels[0], marker='o')
    plt.plot(x_values, behav_subset, label=legend_labels[1], marker='x')
    
    # Set labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel1)
    if behav_var_name:
        plt.ylabel(ylabel1)  # Decoded variable label
        plt.twinx().set_ylabel(behav_var_name)  # Behavioral variable label
    plt.title(title)
    
    # Add legend
    plt.legend()
    
    # Append session information to title if provided
    if session_idx is not None:
        plt.title(f"{title} - Session {session_idx}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        # Ensure the save directory exists
        filename = f"decoded_var_{behav_var_name}.png"
        full_save_path = os.path.join(save_path, f'session_{session_idx}')
        os.makedirs(full_save_path,exist_ok=True)

        # Adjust layout and show the plot
        plt.tight_layout()

        plt.savefig(os.path.join(full_save_path,filename), format='png', dpi=300, bbox_inches='tight')      

        print(f"Plot saved to {full_save_path}")
    
    # Display the plot
    plt.show()

def get_H_over_lap(H=None, true_angle=None):
    """
    Computes the lap number for each H value based on true_angle and returns the paired and sorted arrays.
    
    Parameters:
    - H: array-like, shape (n,)
        Array representing the variable to plot over laps.
    - true_angle: array-like, shape (n,)
        Array representing angles in radians at each time point.
    
    Returns:
    - lap_number: numpy.ndarray, shape (n,)
        Array representing the lap index for each H value.
    - sorted_H: numpy.ndarray, shape (n,)
        The H array reordered based on ascending lap_number.
    - sorted_lap_number: numpy.ndarray, shape (n,)
        The lap_number array sorted in ascending order.
    
    Raises:
    - ValueError: If H and true_angle are not provided or have different lengths.
    """
    
    H = np.asarray(H)
    true_angle = np.asarray(true_angle)

    min_length = min(len(H), len(true_angle))

    H = H[:min_length]
    true_angle = true_angle[:min_length]
    
    if H.ndim != 1 or true_angle.ndim != 1:
        raise ValueError("Both H and true_angle must be 1-dimensional arrays.")
    
    # Compute lap_number as true_angle divided by 360
    lap_number = true_angle / (2*np.pi)
    
    # Sort lap_number and reorder H accordingly
    sorted_indices = np.argsort(lap_number)
    sorted_lap_number = lap_number[sorted_indices]
    sorted_H = H[sorted_indices]
    
    return lap_number, sorted_H, sorted_lap_number


# def plot_Hs_over_laps_scrollable_with_plotly(est_H=None, decode_H=None, lap_number=None, 
#                                              session_idx=None, session=None, save_path=None, tag=None, window_size=None):
#     """
#     Plots two H values against lap numbers with interactivity using Plotly,
#     and saves the plot as an HTML file.
#     """

#     # Input validation
#     if est_H is None or decode_H is None or lap_number is None:
#         raise ValueError("est_H, decode_H, and lap_number must all be provided.")
    
#     # Ensure inputs are NumPy arrays
#     est_H = np.asarray(est_H)
#     decode_H = np.asarray(decode_H)
#     lap_number = np.asarray(lap_number)

#     # Trimming to the shortest length
#     min_length = min(len(est_H), len(decode_H), len(lap_number))
#     est_H = est_H[:min_length]
#     decode_H = decode_H[:min_length]
#     lap_number = lap_number[:min_length]
    
#     # Create interactive traces
#     trace_est_H = go.Scatter(
#         x=lap_number,
#         y=est_H,
#         mode='lines+markers',
#         name='est_H',
#         line=dict(color='blue'),
#     )
#     trace_decode_H = go.Scatter(
#         x=lap_number,
#         y=decode_H,
#         mode='lines+markers',
#         name='decode_H',
#         line=dict(color='red'),
#     )

#     # Create layout
#     layout = go.Layout(
#         title="Interactive H Plot with Scrollable Laps",
#         xaxis=dict(
#             title="Lap Number",
#             rangeslider=dict(visible=True),  # Add range slider for horizontal scrolling
#         ),
#         yaxis=dict(title="H Value"),
#         legend=dict(orientation="h", x=0.5, y=-0.2, xanchor="center"),
#     )

#     fig = go.Figure(data=[trace_est_H, trace_decode_H], layout=layout)

#     # Save or show the plot
#     if save_path:
#         # Ensure directory exists
#         os.makedirs(save_path, exist_ok=True)
#         # Save as HTML
#         filename = f"session_{session_idx}/{save_path}/h_over_laps_interactive_{tag}.html" if tag else f"{save_path}/h_over_laps_interactive.html"
#         plot(fig, filename=filename, auto_open=False)
#         print(f"Saved interactive plot to {filename}")
#     else:
#         # Display the interactive plot in the browser
#         plot(fig)

def plot_and_save_behav_vars(binned_hipp_angle=None, binned_true_angle=None, binned_est_gain=None,
                             save_dir=None, session_idx=None):
    """
    Plots binned hippocampal angle, true angle, and estimated gain on the same figure and saves the plot.

    Parameters:
    - binned_hipp_angle_rad (array-like): Binned hippocampal angles in radians.
    - binned_true_angle_rad (array-like): Binned true angles in radians.
    - binned_est_gain (array-like): Binned estimated gain values.
    - save_dir (str): Base directory where the plot will be saved.
    - session_idx (int): Index of the current session, used to create a subdirectory.
    - behav_var_name (str): Name of the behavioral variable, used in the filename.

    Returns:
    - None
    """

    # Ensure all input arrays have the same length
    if not (len(binned_hipp_angle) == len(binned_true_angle) == len(binned_est_gain)):
        raise ValueError("All input arrays must have the same length.")

    # Define the filename and full save path
    filename = f"behav_vars.png"
    full_save_path = os.path.join(save_dir, f'session_{session_idx}')
    
    # Create the directory if it doesn't exist
    os.makedirs(full_save_path, exist_ok=True)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define the x-axis as bin indices
    length = 100
    x = np.arange(100)
    
    # Plot each behavioral variable
    ax.plot(x, binned_hipp_angle[100:200], label='Hippocampal Angle (rad)', color='blue', linewidth=1.5)
    ax.plot(x, binned_true_angle[100:200], label='True Angle (rad)', color='green', linewidth=1.5)
    ax.plot(x, binned_est_gain[100:200], label='Estimated Gain', color='red', linewidth=1.5)
    

    ax.set_xlabel('time (s)', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title(f'Behavioral Variables for Session {session_idx}', fontsize=16)
    

    ax.legend(fontsize=12)
   
    ax.grid(True, linestyle='--', alpha=0.6)
  
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(full_save_path, filename)
    fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.close(fig)

def low_pass_filter(angles=None, cutoff_frequency=0.1, filter_order=3):
    """
    Apply a low-pass Butterworth filter to smooth the `decoded_angles` array.

    Parameters:
    - decoded_angles (np.array): The 1D array of angles sampled at 1 Hz.
    - cutoff_frequency (float): The cutoff frequency for the low-pass filter (default: 0.1 Hz).
    - filter_order (int): The order of the Butterworth filter (default: 3).

    Returns:
    - np.array: The smoothed array.
    """
    # Sampling frequency
    fs = 1  # Hz
    
    # Normalize cutoff frequency with respect to Nyquist frequency (fs/2)
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_frequency / nyquist

    #Butterworth low-pass filter
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    
    # Apply the filter using filtfilt to preserve phase
    smoothed_angles = filtfilt(b, a, angles)
    
    return smoothed_angles

def compute_moving_average(data=None, window_size=None):
    """
    Computes the moving average of a 1D NumPy array.

    Parameters:
    - data (np.ndarray): Input data array.
    - window_size (int): Size of the moving window.

    Returns:
    - np.ndarray: Moving averaged data.
    """
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')












