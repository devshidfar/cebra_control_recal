import sys
import os
import cebra
sys.path.append("/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/spud_code/shared_scripts")
import manifold_fit_and_decode_fns_custom as mff
import fit_helper_fns_custom as fhf
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
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from ripser import ripser
from persim import plot_diagrams
import umap.umap_ as umap
from sklearn.manifold import TSNE

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
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from ripser import ripser
    from persim import plot_diagrams

    # Create a folder for the session and dimension
    session_folder = os.path.join(results_save_path,"persistent_homology", f"session_{session_idx}", f"dimension_{dimension}")
    os.makedirs(session_folder, exist_ok=True)
    
    # Compute persistent homology
    ripser_result = ripser(embeddings, maxdim=1)
    diagrams = ripser_result['dgms']
    
    # Save the persistent homology results manually (as the shapes are inconsistent)
    homology_filename = os.path.join(session_folder, f"ripser_result_dimension_{dimension}.npz")
    
    # Save individual diagrams as arrays
    np.savez(homology_filename, H0=diagrams[0], H1=diagrams[1])
    print(f"Saved persistent homology results (H0 and H1) to {homology_filename}")
    
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
    from sklearn.neighbors import NearestNeighbors
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


def fit_spud_to_cebra(embeddings, ref_angles=None,hippocampal_angle_origin=None,session_idx=None, 
                       session=None, results_save_path=None,fit_params=None,dimension_3d=None):
#     # Set up the fit parameters, taken base from Chaudhuri et al.

    print(f"dense points: {embeddings}")
    # Create fitter object
    fitter = mff.PiecewiseLinearFit(embeddings, fit_params)
    # Get initial knots
    print("made it here")
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
    fitter.fit_data(curr_fit_params)

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

    # knot1 = final_knots[idx1]
    # knot2 = final_knots[idx2]
    # print(f"Knot {max_dist_idx}: {knot1}")
    # print(f"Knot {max_dist_idx + 1}: {knot2}")

    
    # if max_dist > 1.5*np.median(knot_dists):
    #     print(f"Removing outlier knot at index {max_dist_idx} with distance {max_dist}")
    #     # Remove the knot
    #     final_knots = np.delete(final_knots, max_dist_idx, axis=0)
    # else:
    #     print("No outlier knot found; proceeding without removing any knots.")

    # # Adjust the number of knots
    # nKnots = final_knots.shape[0]

    # construct the spline
    loop_final_knots = fhf.loop_knots(final_knots)
    loop_final_knots_pre = fhf.loop_knots(final_knots_pre)
    tt, curve = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')
    _, curve_pre = fhf.get_curve_from_knots(loop_final_knots_pre, 'eq_vel')

    print(f"form of tt in fit_spud_to_cebra: {np.max(tt)-np.min(tt)}")
    print(f"form of hippangle in fit_spud_to_cebra: {np.max(ref_angles)-np.min(ref_angles)}")

    if ref_angles is not None and hippocampal_angle_origin is not None:
        # Find the index of the closest hippocampal angle to the desired origin
        origin_idx = np.argmin(np.abs(ref_angles - hippocampal_angle_origin))

        tt = tt * 2*(np.pi)

        # Shift the tt values so that the origin is aligned with tt = 0
        tt_shifted = np.mod(tt - tt[origin_idx], 2 * np.pi)  # Keep tt in the [0, 2*pi] range

        tt_diff = np.diff(tt_shifted)
        angle_diff = np.diff(ref_angles)

        # Check if the signs of the slopes match, if not, reverse the tt and curve
        if np.sign(tt_diff[0]) != np.sign(angle_diff[0]):
            print("Reversing spline direction to align with hippocampal angles")
            tt_shifted = np.flip(tt_shifted)
            curve = np.flip(curve, axis=0)
            curve_pre = np.flip(curve_pre, axis=0)
        
        # Reparametrize the spline so that the new tt is aligned with hippocampal_angle_origin
        print(f"Reparametrized spline with origin at hippocampal angle: {hippocampal_angle_origin} (index {origin_idx})")
        
        tt = tt_shifted

    if ref_angles is not None:
        # Use the `decode_from_spline_fit` function (similar to `decode_from_passed_fit`)
        decoded_angles, mse = mff.decode_from_passed_fit(embeddings, tt[:-1], curve[:-1], ref_angles)
        print(f"Decoded circular MSE: {mse}")
        return curve, curve_pre, tt, decoded_angles, mse
    
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
    

def create_rotating_3d_plot(embeddings_3d=None, session=None, behav_var=None, name_behav_var=None, anim_save_path=None, save_anim=None, principal_curve=None, tt=None, num_labels=10,mean_dist=None):
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
            f'Mean Distance from spline: {mean_dist:.2f}',
            fontsize=12,
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

def apply_cebra(neural_data,output_dimensions,max_iterations=None,batch_size=None,temperature=1):
    model = cebra.CEBRA(output_dimension=output_dimensions, max_iterations=1000, batch_size=128,temperature=2)
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
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=10, c=(hipp_angle_binned % 360), cmap='viridis')
    plt.title('t-SNE Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hippocampal Angle (degrees)')
    plt.show()
    print("Displayed t-SNE plot.")

    return


def calculate_single_H(principal_curve=None, tt=None, embeddings=None, t0=None, t1=None, true_angle=None):
    
    # Extract embeddings at t0 and t1
    embedding_t0 = embeddings[t0].reshape(1, -1)  # Reshape to (1, n) for knn
    embedding_t1 = embeddings[t1].reshape(1, -1)

    # Use get_closest_manifold_coords to find nearest manifold points
    input_coords_0,dists_from_mani_0,tt_index0 = fhf.get_closest_manifold_coords(principal_curve, tt, embedding_t0, return_all = True)
    input_coords_0,dists_from_mani_0,tt_index1 = fhf.get_closest_manifold_coords(principal_curve, tt, embedding_t1, return_all = True)
    #now find euclidean nearest point to spline from each embedding point

    #will get tt_index1,tt_index0

    H = (tt[tt_index1] - tt[tt_index0])/(true_angle[t1]-true_angle[t0])

    return H


def calculate_over_experiment_H(principal_curve=None, tt=None, embeddings=None,true_angle=None):
    
    num_avg_over = 4
    spacing = 2
    H_list = []
    for i in range(((len(embeddings)-1)-num_avg_over)//spacing):

        H_temp_list = []
        t0 = i + spacing
        for j in range(num_avg_over):
            t1 = i + j + spacing
            H_temp = calculate_single_H(principal_curve, tt, embeddings, t0, t1, true_angle)
            H_temp_list.append(H_temp)

        H_temp = np.mean(H_temp_list)
        H_list.append(H_temp)
    
    return np.array(H_list)

def plot_decode_H_vs_true_H(est_H=None, decode_H=None, session_idx=None, session=None, save_path=None):
    """
    Plots est_gain and decode_gain using array indices as time points.

    Parameters:
    - est_gain (np.ndarray): Array of estimated gain values.
    - decode_gain (np.ndarray): Array of decoded gain values.
    - session_idx (int, optional): Session index for labeling purposes.
    - session (object, optional): Session object containing metadata.
    - save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Find the overlapping range of indices
    min_length = min(len(est_H), len(decode_H))

    # Trim est_gain and decode_gain to the overlapping range
    est_gain_trimmed = est_H[:min_length]
    decode_gain_trimmed = decode_H[:min_length]
    times = np.arange(min_length)  # Time is just the index

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(times, est_gain_trimmed, label='Estimated Gain', color='blue')
    plt.plot(times, decode_gain_trimmed, label='Decoded Gain', color='red', alpha=0.7)

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Gain', fontsize=14)
    title = 'FT gain vs spline decoded Gain'
    if session_idx is not None and session is not None:
        title += f'\nSession {session_idx}: Rat {session.rat}, Day {session.day}, Epoch {session.epoch}'
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        save_path = f'{save_path}/h_plots/session_{session_idx}'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/h_est_vs_decode.png", dpi=300)
        plt.close()
       
        print(f"Saved plot to {save_path}")
    else:
        plt.show()



        




