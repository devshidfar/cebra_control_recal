import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.spatial.transform import Rotation as R
import ot  # POT: Python Optimal Transport
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

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
    cost_matrix = np.linalg.norm(source_cloud[:n, None] - target_cloud[:n], axis=2)
    weights_source = np.ones(n) / n
    weights_target = np.ones(n) / n
    emd_value = ot.emd2(weights_source, weights_target, cost_matrix)  # EMD squared
    return emd_value

def compute_partial_emd(source_cloud, target_cloud, mass_ratio=None):
    """
    Compute partial Earth Mover's Distance (EMD) between two 3D point clouds 
    of different sizes using POT (Python Optimal Transport).
    
    Parameters:
        source_cloud (np.ndarray): Shape (N, 3) array of source points.
        target_cloud (np.ndarray): Shape (M, 3) array of target points.
        reg (float): Regularization term (0 for standard EMD, positive for Sinkhorn relaxation).
        mass_ratio (float): Ratio of total mass to be transported (None means automatic detection).
    
    Returns:
        float: Partial EMD (Wasserstein distance).
    """
    N, M = len(source_cloud), len(target_cloud)

    # Compute pairwise Euclidean cost matrix
    cost_matrix = ot.dist(source_cloud,target_cloud,metric='euclidean')

    # Define uniform weights, allowing partial matching
    weights_source = np.ones(N) / N
    weights_target = np.ones(M) / M

    # Compute Partial Wasserstein Distance (partial EMD)
    # mass_ratio = (min(N,M)) /(max(N,M))
    # mass_ratio = min(np.sum(weights_source), np.sum(weights_target))
    # transport_plan = ot.partial.partial_wasserstein(weights_source, weights_target, cost_matrix, m=mass_ratio)
    # emd_value =  np.sum(transport_plan * cost_matrix)
    emd_value = ot.emd2(weights_source,weights_target,cost_matrix)
    # SWD = ot.sliced_wasserstein_distance(source_cloud, target_cloud, n_projections=50)

    # print(f"Sliced Wasserstein Distance: {SWD:.4f}")

    return emd_value

def find_best_rotation(source_cloud, target_cloud, num_trials=100):
    """Find the rotation that minimizes 3D EMD between a source and target cloud."""
    best_rotation = None
    min_distance = float('inf')
    best_cloud = None

    for _ in range(num_trials):
        rotation = random_rotation()
        rotated_cloud = rotate_embeddings(source_cloud, rotation)
        distance = compute_partial_emd(rotated_cloud, target_cloud)
    
        if distance < min_distance:
            min_distance = distance
            best_rotation = rotation
            best_cloud = rotated_cloud

    return best_rotation, best_cloud, min_distance

def plot_point_clouds_interactive(clouds, session_num, save_dir, title="Point Cloud Visualization"):
    """
    Create an interactive 3D plot of multiple point clouds using Plotly and save it as an HTML file.
    """
    fig = go.Figure()
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
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"session_{session_num}.html")
    fig.write_html(save_path)
    print(f"Interactive plot saved to {save_path}")

# -----------------------
# MAIN SCRIPT
# -----------------------
if __name__ == "__main__":
    # Set mode to either "within" or "across"
    # "within": compare multiple rotated clouds within each trial of a single session.
    # "across": take one trial (e.g., trial 0) from each session and compare across sessions.
    mode = "across"  # or "within"

    # Load embeddings
    file_path = '/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results/trial_type_test/trial_type_test_all_sessions_data.mat'
    mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    
    trials = [mat_data[f'full_trial_{i}'] for i in range(1, 5)]
    sessions = [trial.sessions for trial in trials]
    # embeddings_list[i][k] is the embeddings_3d for session i, trial k.
    embeddings_list = [[session.embeddings_3d for session in s] for s in sessions]


    save_dir = "wasserstein_plots"
    os.makedirs(save_dir, exist_ok=True)
    num_trials = 15  # Number of sessions/trials to compare

    if mode == "within":
        # Within-session comparison: For each trial in a given session, compare rotated clouds from different sources.
        # Here, we assume session 0 is the "target" and sessions 1...num_trials-1 are the sources.
        for k in range(2, 10):
            target_cloud = embeddings_list[0][k]
            source_clouds = [embeddings_list[i][k] for i in range(1, num_trials)]
    
            N = source_clouds[0].shape[0]
            best_rotation = np.zeros((num_trials - 1, 3, 3))
            best_rotated_cloud = np.zeros((num_trials - 1, N, 3))
            min_distance = np.zeros(num_trials - 1)
    
            for i in range(num_trials - 1):
                best_rotation[i], best_rotated_cloud[i], min_distance[i] = find_best_rotation(source_clouds[i], target_cloud)
    
            # Plot the best rotated cloud for this trial alongside the target
            best_idx = np.argmin(min_distance)
            plot_point_clouds_interactive(
                clouds=[best_rotated_cloud[best_idx], target_cloud],
                session_num=k,
                save_dir=save_dir,
                title=f"Within Session Trial {k}: Best Alignment (Min EMD: {min(min_distance):.3g})"
            )
            print(f"Trial {k} (within session): min_distance per source = {min_distance}, overall best = {min(min_distance)}")
    
            # --- Compute and save pairwise distance matrix for this trial ---
            n_sources = best_rotated_cloud.shape[0]
            pairwise_matrix = np.zeros((n_sources, n_sources))
            for i in range(n_sources):
                for j in range(n_sources):
                    pairwise_matrix[i, j] = compute_emd_3d(best_rotated_cloud[i], best_rotated_cloud[j])
    
            # Save pairwise distance matrix to a MAT file for this trial.
            mat_save_path = os.path.join(save_dir, f"pairwise_distance_trial_{k}.mat")
            savemat(mat_save_path, {'pairwise_distance_matrix': pairwise_matrix})
            print(f"Saved pairwise distance matrix for trial {k} to {mat_save_path}")
    
    elif mode == "across":
        # Across-session comparison: Compare one trial (e.g., trial 0) across sessions.
        start_point = 0
        target_cloud = embeddings_list[0][7]  # Take the {start_point} trial from session  as the target
        source_clouds = [embeddings_list[0][i] for i in range(start_point, num_trials)]
    
        N = source_clouds[0].shape[0]
        best_rotation = np.zeros((num_trials-start_point, 3, 3))
        best_rotated_cloud = []
        min_distance = np.zeros(num_trials-start_point)
    
        for i in range(num_trials-start_point):
            best_rotation[i], best_rotated_cloud_temp, min_distance[i] = find_best_rotation(source_clouds[i], target_cloud)
            best_rotated_cloud.append(best_rotated_cloud_temp)
            plot_point_clouds_interactive(
                clouds=[best_rotated_cloud_temp, target_cloud],
                session_num=f"across_{i+1}",
                save_dir=save_dir,
                title=f"Across Session: Source Session {i+1} vs Target Session 0 (Min EMD: {min_distance[i]:.3g})"
            )
            print(f"Across session: For source session {i+1}, min_distance = {min_distance[i]}")
    
        # Create an overall pairwise distance matrix across sessions.
        n_sources = best_rotation.shape[0]
        pairwise_matrix = np.zeros((n_sources, n_sources))
        for i in range(n_sources):
            for j in range(n_sources):
                pairwise_matrix[i, j] = compute_partial_emd(best_rotated_cloud[i], best_rotated_cloud[j])
    
        overall_save_path = os.path.join(save_dir, "across_sessions_pairwise_distance.mat")
        savemat(overall_save_path, {'pairwise_distance_matrix': pairwise_matrix})
        print(f"Across-session pairwise distance matrix saved to {overall_save_path}")

        backup_path = file_path.replace(".mat", "_backup.mat")  # Create a backup file
        savemat(backup_path, mat_data)  # Save original file as a backup

        mat_data['pairwise_EMD_distance_matrix'] = pairwise_matrix
        mat_data['sesions_for_EMD'] = [trials[0].sessions[i].session_idx for i in range(start_point,num_trials)]
        # Save the modified data to the original file
        savemat(backup_path, mat_data)

        print(f"Backup saved at: {backup_path}")
        print(f"Updated .mat file saved at: {file_path}")
    
        # Additionally, create an interactive Plotly figure comparing the min distances.
        fig = go.Figure()
        sessions_list = list(range(1, num_trials))  # source sessions 1 to num_trials-1
        fig.add_trace(go.Scatter(
            x=sessions_list,
            y=min_distance,
            mode='markers+lines',
            marker=dict(size=8, color='blue'),
            name='Min Wasserstein Distance'
        ))
        fig.update_layout(
            title="Across-Session Comparison (Trial 0)",
            xaxis_title="Source Session (compared to Target Session 0)",
            yaxis_title="Minimum Wasserstein (EMD²)"
        )
        overall_plot_path = os.path.join(save_dir, "across_sessions_comparison.html")
        fig.write_html(overall_plot_path)
        print(f"Across-session comparison plot saved to {overall_plot_path}")
    
    else:
        print("Invalid mode. Please set mode to either 'within' or 'across'.")
