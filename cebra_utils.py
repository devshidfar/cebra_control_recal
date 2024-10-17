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
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from ripser import ripser
from persim import plot_diagrams

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

def fit_spud_to_cebra(embeddings_3d, ref_angles=None, nKnots=20, knot_order='nearest', penalty_type='curvature', length_penalty=5,hippocampal_angle_origin=None):
    # Set up the fit parameters, taken base from Chaudhuri et al.
    fit_params = {
        'dalpha': 0.005,
        'knot_order': knot_order,
        'penalty_type': penalty_type,
        'nKnots': nKnots,
        'length_penalty': length_penalty,
        'curvature_coeff': 5,
        'len_coeff': 1
    }


    lof = LocalOutlierFactor(n_neighbors=5, contamination = 0.1)
    is_inlier = lof.fit_predict(embeddings_3d) == 1
    embeddings_3d_inliers = embeddings_3d[is_inlier]

    if ref_angles is not None:
        # Use reference angles to guide the fitting process
        ref_angles_inliers = ref_angles[is_inlier]

    tree = KDTree(embeddings_3d_inliers)
    dist, _ = tree.query(embeddings_3d_inliers, k=6)  # Get distances to the 5 nearest neighbors
    avg_dist = np.mean(dist[:, 1:], axis=1)  # Average distance to nearest neighbors (excluding self)

    # Apply a threshold to remove points that are too far from their neighbors
    density_threshold = np.percentile(avg_dist, 75)  # Adjust threshold as needed (e.g., 75th percentile)
    dense_points = embeddings_3d_inliers[avg_dist < density_threshold]
    
    # Create fitter object
    fitter = mff.PiecewiseLinearFit(dense_points, fit_params)
    # Get initial knots
    unord_knots = fitter.get_new_initial_knots()
   
    init_knots = fitter.order_knots(unord_knots, method=fit_params['knot_order'])
    
    # Fit the data
    curr_fit_params = {'init_knots': init_knots, 'penalty_type': fit_params['penalty_type'], 'len_coeff': fit_params['len_coeff'],'curvature_coeff': fit_params['curvature_coeff']}
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

    # print(f"The largest distance between consecutive knots is {max_dist}, "
    #       f"between knot {max_dist_idx} and knot {max_dist_idx + 1}")

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

    if ref_angles is not None and hippocampal_angle_origin is not None:
        # Find the index of the closest hippocampal angle to the desired origin
        origin_idx = np.argmin(np.abs(ref_angles_inliers - hippocampal_angle_origin))

        # Shift the tt values so that the origin is aligned with tt = 0
        tt_shifted = np.mod(tt - tt[origin_idx], 2 * np.pi)  # Keep tt in the [0, 2*pi] range

        tt_diff = np.diff(tt_shifted)
        angle_diff = np.diff(ref_angles_inliers)

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
        decoded_angles, mse = mff.decode_from_passed_fit(embeddings_3d_inliers, tt[:-1], curve[:-1], ref_angles_inliers)
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
#             anim.save(f"{anim_save_path}{name_behav_var}.gif", writer='pillow', fps=30)
#         else:
#             plt.show()

#     return anim

def create_rotating_3d_plot(embeddings_3d, session, behav_var, name_behav_var, anim_save_path, save_anim, principal_curve=None, tt=None, num_labels=10):
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

    tt=tt*2*np.pi

    # Normalize `behav_var` and `tt` to use the same color scale
    norm = plt.Normalize(vmin=np.min(behav_var), vmax=np.max(behav_var))
    cmap = plt.get_cmap('viridis')  # Same color map for both scatter and spline

    # Scatter plot of embeddings using behav_var as color map
    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                         c=behav_var, cmap=cmap, norm=norm, s=5)
    plt.colorbar(scatter, label=f'{name_behav_var}')

    # Plot the principal curve with the same color map, colored by `tt` values
    if principal_curve is not None and tt is not None:
        for i in range(len(principal_curve) - 1):
            ax.plot(principal_curve[i:i+2, 0], principal_curve[i:i+2, 1], principal_curve[i:i+2, 2], 
                    color=cmap(norm(tt[i])), linewidth=2)

        # Add labels at evenly spaced points along the spline
        label_indices = np.linspace(0, len(principal_curve) - 1, num_labels, dtype=int)
        for idx in label_indices:
            x, y, z = principal_curve[idx]
            ax.text(x, y, z, f'tt={tt[idx]:.2f}', color='black', fontsize=8)

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

