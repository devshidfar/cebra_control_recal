"""
File: cebra_analysis_oop.py

Dependencies:
    - cebra
    - numpy, scipy, matplotlib, scikit-learn, etc. 
    - The same dependencies as in your original code.

Example run:
    python run_manifold_H.py
    # or within a Python shell:
    # main()
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
from scipy.io import savemat
from matplotlib.backends.backend_pdf import PdfPages
import cebra
import matplotlib as plt




plt.use("Agg")

# =============== #
# CLASS: CEBRAUtils
# =============== #

class CEBRAUtils:
    """
    Static utility methods for applying CEBRA, computing metrics, plotting, etc.
    These were originally from 'CEBRA_Utils.py'.
    """

    @staticmethod
    def apply_cebra(neural_data=None, output_dimension=3, temperature=1):
        """
        Apply the CEBRA model to 'neural_data' and return the embeddings.
        """
        model = cebra.CEBRA(
            output_dimension=output_dimension,
            max_iterations=1000,
            batch_size=512,
            temperature=temperature
        )
        model.fit(neural_data)
        embeddings = model.transform(neural_data)
        return embeddings

    @staticmethod
    def decode_hipp_angle_spline(
            embeddings=None,
            principal_curve=None,
            tt=None,
            behav_angles=None,
            true_angles=None,
            decode_fn=None
    ):
        """
        Calculate the hippocampal angle decoded from the spline.
        """
        # Example default decode function
        if decode_fn is None:
            from manifold_fit_and_decode_fns_custom import decode_from_passed_fit as default_decode
            decode_fn = default_decode

        decoded_angles, mse = decode_fn(embeddings, tt[:-1], principal_curve[:-1], true_angles)
        return decoded_angles, mse

    @staticmethod
    def window_smooth(data=None, window_size=3):
        """
        Compute finite differences and then the moving average
        of those differences, essentially a smoothing operation.
        """
        diffs = np.diff(data)
        kernel = np.ones(window_size) / window_size
        avg_diffs = np.convolve(diffs, kernel, mode='valid')
        return avg_diffs

    @staticmethod
    def nt_TDA(data, pct_distance=1, pct_neighbors=20, pct_dist=90):
        """
        Outlier removal using pairwise distances 
        and neighbor-based thresholds (TDA-inspired).
        """
        from scipy.spatial import distance_matrix
        from sklearn.neighbors import NearestNeighbors

        distances = distance_matrix(data, data)
        np.fill_diagonal(distances, 10)
        
        # Determine the neighborhood radius for each point
        neighborhood_radius = np.percentile(distances, pct_distance, axis=0)
        neighbor_counts = np.sum(distances <= neighborhood_radius[:, None], axis=1)
        threshold_neighbors = np.percentile(neighbor_counts, pct_neighbors)
        outlier_indices = np.where(neighbor_counts < threshold_neighbors)[0]

        neighbgraph = NearestNeighbors(n_neighbors=5).fit(distances)
        dists, _ = neighbgraph.kneighbors(distances)
        min_distance_to_any_point = np.mean(dists, axis=1)
        distance_threshold = np.percentile(min_distance_to_any_point, pct_dist)
        far_outliers = np.where(min_distance_to_any_point > distance_threshold)[0]

        outlier_indices = np.unique(np.concatenate([outlier_indices, far_outliers]))
        all_indices = np.arange(data.shape[0])
        inlier_indices = np.setdiff1d(all_indices, outlier_indices)
        return inlier_indices

    @staticmethod
    def fit_spud_to_cebra(
            embeddings,
            ref_angle=None,
            session_idx=None,
            session=None,
            results_save_path=None,
            fit_params=None,
            dimension_3d=None,
            verbose=False
    ):
        """
        Fits a custom principal curve (SPUD) to the data in 'embeddings' 
        using 'fit_params' from manifold_fit_and_decode_fns_custom.py.
        """
        import shared_scripts.manifold_fit_and_decode_fns_custom as mff
        import shared_scripts.fit_helper_fns_custom as fhf
        # from real_data import cebra_analysis_oop  # if you separate modules; otherwise ignore

        fitter = mff.PiecewiseLinearFit(embeddings, fit_params)
        unord_knots = fitter.get_new_initial_knots(method='kmedoids')
        init_knots = fitter.order_knots(unord_knots, method=fit_params['knot_order'])

        # Plot initial knots in 3D
        if dimension_3d == 1 and results_save_path:
            plot_save_path = os.path.join(results_save_path, f"initial_knots_session_{session_idx}.png")
            CEBRAUtils.plot_initial_knots(embeddings, init_knots, session_idx, session, save_path=plot_save_path)

        curr_fit_params = {'init_knots': init_knots, **fit_params}
        fitter.fit_data(curr_fit_params, verbose=verbose)
        final_knots = fitter.saved_knots[0]['knots']

        # knot_spacing = np.linalg.norm(final_knots[1:] - final_knots[:-1], axis=1)
        # min_dist = np.min(knot_spacing) if len(knot_spacing) else np.inf
        # #threshold for skipping the session (caused by not enough points and
        # #thus no valid principal curve)
        # threshold = 1e-12
        # if min_dist < threshold:
        #     print(f"[WARNING] Knots are too close: Minimum distance between knots "
        #         f"is {min_dist:.2e}. Skipping this session.")
        #     return None, None, None  # This returns to 'main' to skip
        
        # Possibly remove an outlier knot if it is too far
        segments = np.vstack((final_knots[1:] - final_knots[:-1], final_knots[0] - final_knots[-1]))
        knot_dists = np.linalg.norm(segments, axis=1)
        max_dist_idx = np.argmax(knot_dists)
        max_dist = np.max(knot_dists)
        nKnots = final_knots.shape[0]
        if max_dist_idx < nKnots - 1:
            idx1 = max_dist_idx
            idx2 = max_dist_idx + 1
        else:
            idx1 = nKnots - 1
            idx2 = 0

        if max_dist > 1.5 * np.median(knot_dists):
            final_knots = np.delete(final_knots, idx2, axis=0)

        # Build final spline
        loop_final_knots = fhf.loop_knots(final_knots)
        tt, curve = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')
        _, curve_pre = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')

        # Shift 'tt' to align with reference angles
        if ref_angle and tt:
            tt = tt * 2 * np.pi
            tt_shifted = (tt + ref_angle[3]) % (2 * np.pi)

            # Check the first slope sign
            tt_diff = np.diff(tt_shifted)
            angle_diff = np.diff(ref_angle)
            if np.sign(tt_diff[0]) != np.sign(angle_diff[0]):
                tt_shifted = np.flip(tt_shifted)
                curve = np.flip(curve, axis=0)
            tt = tt_shifted

        return curve, curve_pre, tt

    @staticmethod
    def plot_initial_knots(data_points, init_knots, session_idx, session, save_path=None):
        """
        Plots initial knots in 3D for debugging/visualization.
        """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2],
                   c='gray', s=5, alpha=0.5, label='Data Points')
        ax.scatter(init_knots[:, 0], init_knots[:, 1], init_knots[:, 2],
                   c='red', s=100, marker='^', label='Initial Knots')
        ax.set_title(f'Initial Knots - Session {session_idx}\nRat {session.rat}, '
                     f'Day {session.day}, Epoch {session.epoch}')
        ax.set_xlabel('Embedding Dimension 1')
        ax.set_ylabel('Embedding Dimension 2')
        ax.set_zlabel('Embedding Dimension 3')
        ax.legend()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def mean_dist_to_spline(embeddings=None, principal_curve=None):
        """
        Compute mean distance from the embeddings to the principal_curve.
        """
        from scipy.spatial import KDTree

        def interpolate_principal_curve(principal_curve, points_per_unit_distance=10):
            """
            Utility to do a linear interpolation for the principal_curve.
            """
            import numpy as np
            interpolated_curve = []
            # try:
            for i in range(len(principal_curve) - 1):
                point_start = principal_curve[i]
                point_end = principal_curve[i + 1]
                segment = point_end - point_start
                distance = np.linalg.norm(segment)
                num_points = max(int(points_per_unit_distance * distance), 2)
                interp_points = np.linspace(point_start, point_end, num=num_points, endpoint=False)
                interpolated_curve.append(interp_points)
            interpolated_curve.append(principal_curve[-1].reshape(1, -1))
            interpolated_curve = np.vstack(interpolated_curve)
            return interpolated_curve
            # except ValueError as e:
            #     # Catch "principal curve dne" 
            #     print(f"[WARNING] principal curve failed for session. Exception: {e}")
            #     # Skip this session and continue with the next

        # Interpolate curve
        pc_interp = interpolate_principal_curve(principal_curve)
        if pc_interp:
            tree = KDTree(pc_interp)
            distances, _ = tree.query(embeddings, k=1)
            mean_distance = np.mean(distances)
            return mean_distance

    @staticmethod
    def low_pass_filter(angles=None, cutoff_frequency=0.1, filter_order=3, fs=1):
        """
        Apply a low-pass Butterworth filter to smooth the passed array of angles.
        """
        from scipy.signal import butter, filtfilt

        nyquist = 0.5 * fs
        normalized_cutoff = cutoff_frequency / nyquist
        b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        smoothed_angles = filtfilt(b, a, angles)
        return smoothed_angles

    @staticmethod
    def compute_moving_average(data=None, window_size=None):
        """
        Computes the moving average of a 1D array using a simple rectangular kernel.
        """
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')

    @staticmethod
    def get_var_over_lap(var=None, true_angle=None):
        """
        Computes the lap number for each item in 'var' based on 'true_angle' 
        and returns the paired and sorted arrays.
        """
        min_length = min(len(var), len(true_angle))
        var = var[:min_length]
        true_angle = true_angle[:min_length]
        lap_number = true_angle / (2 * np.pi)
        sorted_indices = np.argsort(lap_number)
        sorted_lap_number = lap_number[sorted_indices]
        sorted_var = var[sorted_indices]
        return lap_number, sorted_var, sorted_lap_number

    @staticmethod
    def save_data_to_csv(data_dict, save_dir, is_list=False):
        """
        Saves the provided data dictionary or list of dictionaries into a CSV file.
        """
        import pandas as pd
        os.makedirs(save_dir, exist_ok=True)
        if is_list:
            df = pd.DataFrame(data_dict)
            csv_file = os.path.join(save_dir, "total_results.csv")
        else:
            df = pd.DataFrame([data_dict])
            csv_file = os.path.join(save_dir, "results.csv")
        df = df.round(2)
        df.to_csv(csv_file, index=False)

    @staticmethod
    def compute_SI_and_plot(
        embeddings=None,
        behav_var=None,
        params=None,
        behav_var_name=None,
        save_dir=None,
        session_idx=None,
        dimensions_3=False,
        pdf=None,
        num_used_clusters=None,
    ):
        """
        Computes the Structure Index (SI) using 
        real_data.SI_code.structure_index's compute_structure_index method,
        and creates a 3-panel figure with adjacency matrix and graph.
        """
        import sys
        import os
        sys.path.append("/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/")
        from SI_code.structure_index import compute_structure_index, draw_graph
        import matplotlib.pyplot as plt
        import numpy as np

        SI, binLabel, overlap_mat, sSI = compute_structure_index(embeddings, behav_var, **params)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 3D scatter if needed
        if dimensions_3:
            from mpl_toolkits.mplot3d import Axes3D
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
        else:
            # 2D scatter
            scatter = ax[0].scatter(embeddings[:, 0], embeddings[:, 1], c=behav_var, cmap='inferno_r', vmin=0, vmax=1)
            cbar = fig.colorbar(scatter, ax=ax[0], anchor=(0, 0.3), shrink=0.8, ticks=[0, 0.5, 1])
            cbar.set_label('radius', rotation=90)
            ax[0].set_title(f'Embeddings with {behav_var_name} feature', size=16)
            ax[0].set_xlabel('Dim 1', size=14)
            ax[0].set_ylabel('Dim 2', size=14)

        matshow = ax[1].matshow(overlap_mat, vmin=0, vmax=0.5, cmap=plt.cm.viridis)
        ax[1].xaxis.set_ticks_position('bottom')
        cbar = fig.colorbar(matshow, ax=ax[1], anchor=(0, 0.2), shrink=1, ticks=[0, 0.25, 0.5])
        cbar.set_label('overlap score', rotation=90, fontsize=14)
        ax[1].set_title('Adjacency matrix', size=16)
        ax[1].set_xlabel('bin-groups', size=14)
        ax[1].set_ylabel('bin-groups', size=14)

        draw_graph(overlap_mat, ax[2], node_cmap=plt.cm.inferno_r, edge_cmap=plt.cm.Greys,
                   node_names=np.round(binLabel[1][:, 0, 1], 2))
        ax[2].set_xlim(1.2 * np.array(ax[2].get_xlim()))
        ax[2].set_ylim(1.2 * np.array(ax[2].get_ylim()))
        ax[2].set_title('Directed graph', size=16)
        ax[2].text(0.98, 0.05, f"SI: {SI:.2f}, #UsedClusters: {num_used_clusters}", 
                   ha='right', va='bottom', transform=ax[2].transAxes, fontsize=25)

        os.makedirs(save_dir, exist_ok=True)
        filename = f"SI_{behav_var_name}.png"
        fig.savefig(os.path.join(save_dir, filename), format='png', dpi=300, bbox_inches='tight')
        if pdf:
            pdf.savefig(fig)
        plt.close(fig)
        return SI

    @staticmethod
    def plot_and_save_behav_vars(
        binned_hipp_angle=None,
        binned_true_angle=None,
        binned_est_gain=None,
        save_dir=None,
        session_idx=None,
    ):
        """
        Plots binned hippocampal angle, true angle,
        and estimated gain on the same figure and saves the plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if not (len(binned_hipp_angle) == len(binned_true_angle) == len(binned_est_gain)):
            raise ValueError("All input arrays must have the same length.")

        filename = "behav_vars.png"
        full_save_path = os.path.join(save_dir)
        os.makedirs(full_save_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        length = 100  # user-defined segment to look at
        x = np.arange(length)
        ax.plot(x, binned_hipp_angle[100:200], label='Hippocampal Angle (rad)', color='blue', linewidth=1.5)
        ax.plot(x, binned_true_angle[100:200], label='True Angle (rad)', color='green', linewidth=1.5)
        ax.plot(x, binned_est_gain[100:200], label='Estimated Gain', color='red', linewidth=1.5)

        ax.set_xlabel('time (s)', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(f'Behavioral Variables for Session {session_idx}', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(full_save_path, filename)
        fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_decoded_var_and_true(
        decoded_var,
        behav_var,
        indices=range(200),
        xlabel="Time (s)",
        ylabel1="H Decoded from Manifold",
        ylabel2="Ground Truth H",
        title="H Decoded from Manifold vs Ground Truth H",
        legend_labels=None,
        save_path=None,
        figsize=(12, 6),
        session_idx=None,
        behav_var_name=None,
        pdf=None,
    ):
        """
        Plots decoded_var and behav_var over specified indices
        and optionally saves to a specified path.
        """
        import matplotlib.pyplot as plt

        indices = np.arange(0, len(decoded_var))
        if len(decoded_var) != len(behav_var):
            raise ValueError("decoded_var and behav_var must have the same length")

        decoded_subset = [decoded_var[i] for i in indices]
        behav_subset = [behav_var[i] for i in indices]
        x_values = list(indices)

        plt.figure(figsize=figsize)
        ax = plt.gca()

        if legend_labels:
            plt.plot(x_values, decoded_subset, label=legend_labels[0], color='red', linestyle='-')
            plt.plot(x_values, behav_subset, label=legend_labels[1], color='black', linestyle='-')
        else:
            plt.plot(x_values, decoded_subset, color='red', linestyle='-')
            plt.plot(x_values, behav_subset, color='black', linestyle='-')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel1)
        plt.title(title)
        ax.grid(False)
        if legend_labels and len(legend_labels) == 2:
            plt.legend(loc="best")
        if session_idx is not None:
            plt.title(f"{title} - Session {session_idx}")

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"decoded_var_{behav_var_name}.png" if behav_var_name else "decoded_var_plot.png"
            plt.savefig(os.path.join(save_path, filename), format='png', dpi=300, bbox_inches='tight')
            if pdf:
                pdf.savefig()
        plt.show()

    @staticmethod
    def plot_spatial_spectrogram(
        H=None,
        lap_numbers=None,
        save_path=None,
        behav_var_name=None,
        num_segments_per_lap=10,
    ):
        """
        Plots a standard (matplotlib) spectrogram of H as a function of lap_number.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import signal
        from scipy.interpolate import interp1d

        if H is None or lap_numbers is None:
            raise ValueError("H and lap_numbers must both be provided.")

        # Check spacing of lap_numbers
        differences = np.diff(lap_numbers)
        if not np.allclose(differences, differences[0], atol=1e-6):
            # Resample
            num_samples = len(lap_numbers)
            lap_numbers_uniform = np.linspace(lap_numbers.min(), lap_numbers.max(), num_samples)
            interp_func = interp1d(lap_numbers, H, kind="linear", fill_value="extrapolate")
            H_uniform = interp_func(lap_numbers_uniform)
            lap_numbers = lap_numbers_uniform
            H = H_uniform

        dx = lap_numbers[1] - lap_numbers[0]
        fs = 1 / dx

        average_samples_per_lap = len(H) / (np.max(lap_numbers) - np.min(lap_numbers))
        nperseg = int(average_samples_per_lap)
        nperseg = max(1, nperseg)
        noverlap = nperseg // 2

        frequencies, positions, Sxx = signal.spectrogram(
            H, fs=fs, window="hann", 
            nperseg=nperseg, 
            noverlap=noverlap, 
            scaling="density", 
            mode="magnitude"
        )

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(positions, frequencies, Sxx, shading="gouraud", cmap="viridis")
        plt.title("Spatial Spectrogram of H over Lap Numbers")
        plt.ylabel("Spatial Frequency [cycles per lap]")
        plt.xlabel("Lap Number")
        plt.colorbar(label="Intensity")
        plt.tight_layout()

        if save_path and behav_var_name:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"{behav_var_name}.png")
            plt.savefig(file_path)
        plt.show()
        plt.close()

        # Plot the dominant frequencies
        dominant_frequencies = frequencies[np.argmax(Sxx, axis=0)]
        plt.figure(figsize=(10, 4))
        plt.plot(positions, dominant_frequencies, marker="o")
        plt.xlabel("Lap Number")
        plt.ylabel("Dominant Frequency [cycles per lap]")
        plt.title("Dominant Spatial Frequency Across Laps")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_spatial_spectrogram_interactive(
        H=None,
        lap_numbers=None,
        save_path=None,
        behav_var_name=None,
        session_idx=None,
        session=None,
        tag=None,
        additional_annotations=None,
        num_segments_per_lap=10
    ):
        """
        Creates an interactive spectrogram (Plotly) of H over lap_number.
        """
        import numpy as np
        from scipy import signal
        from scipy.interpolate import interp1d
        import plotly.graph_objs as go
        from plotly.offline import plot

        if H is None or lap_numbers is None:
            raise ValueError("H and lap_numbers must both be provided.")

        differences = np.diff(lap_numbers)
        if not np.allclose(differences, differences[0], atol=1e-6):
            num_samples = len(lap_numbers)
            lap_numbers_uniform = np.linspace(lap_numbers.min(), lap_numbers.max(), num_samples)
            interp_func = interp1d(lap_numbers, H, kind="linear")
            H_uniform = interp_func(lap_numbers_uniform)
            lap_numbers = lap_numbers_uniform
            H = H_uniform

        dx = lap_numbers[1] - lap_numbers[0]
        fs = 1 / dx
        average_samples_per_lap = len(H) / (np.max(lap_numbers) - np.min(lap_numbers))
        nperseg = int(average_samples_per_lap / 2)
        nperseg = max(1, nperseg)
        noverlap = nperseg // 2

        frequencies, positions, Sxx = signal.spectrogram(
            H, 
            fs=fs, 
            window="hann", 
            nperseg=nperseg, 
            noverlap=noverlap, 
            scaling="density", 
            mode="magnitude"
        )

        heatmap = go.Heatmap(
            z=Sxx,
            x=positions,
            y=frequencies,
            colorscale="Viridis",
            colorbar=dict(title="Intensity"),
        )

        base_title = f"Spatial Spectrogram of {behav_var_name} Over Laps"
        if session_idx is not None and session is not None:
            base_title += f"<br>Session {session_idx}: Rat {session.rat},"
            " Day {session.day}, Epoch {session.epoch}"
            if tag:
                base_title += f", Tag {tag}"

        layout = go.Layout(
            title=base_title,
            xaxis=dict(title="Lap Number"),
            yaxis=dict(title="Spatial Frequency [cycles per lap]"),
        )

        fig = go.Figure(data=[heatmap], layout=layout)
        if additional_annotations:
            annotation_text = "<br>".join([f"{key}: {val}" for key, val in additional_annotations.items()])
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                text=annotation_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                opacity=0.8,
            )

        if save_path and behav_var_name:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"{behav_var_name}_"
                                     "spectrogram.html")
            plot(fig, filename=file_path, auto_open=False)
        else:
            plot(fig)


# =============== #
# CLASS: CEBRAAnalysis
# =============== #

class CEBRAAnalysis:
    """
    Encapsulates the main analysis steps from your 'main.ipynb',
    using the utility methods from CEBRAUtils.
    """

    def __init__(self, session_choose=True, max_num_reruns=1):
        """
        Loads data, sets up configuration, etc.
        """
        import scipy.io

        # --- Load data ---
        path_flow = os.path.join(
            '/Users/devenshidfar/Desktop/Masters/NRSC_510B/',
            'cebra_control_recal/mat_code_and_data/',
            'data/NN_opticflow_dataset.mat'
        )
        data_flow = scipy.io.loadmat(
            path_flow, 
            squeeze_me=True, 
            struct_as_record=False
        )
        self.expt_optic_flow = data_flow['expt']

        path_landmark = os.path.join(
            '/Users/devenshidfar/Desktop/Masters/NRSC_510B/',
            'cebra_control_recal/mat_code_and_data/',
            'data/expt_landmark.mat'
        )
        data_landmark = scipy.io.loadmat(path_landmark, squeeze_me=True, struct_as_record=False)
        self.expt_landmark = data_landmark['expt']

        print(f"Optic Flow expt shape: {self.expt_optic_flow.shape}")
        print(f"Landmark expt shape: {self.expt_landmark.shape}")

        # --- Config parameters ---
        self.session_choose = session_choose
        if self.session_choose:
            self.landmark_sessions = []
            self.optic_flow_sessions = []
        else:
            self.landmark_num_trials = 65
            self.landmark_control_point = 70
            self.optic_flow_num_trials = 72
            self.optic_flow_control_point = 41


            self.landmark_control_point = self.landmark_control_point - 1
            self.optic_flow_control_point = self.optic_flow_control_point - 1

        self.max_num_reruns = max_num_reruns
        self.save_folder = 'tt_test'

        # List of experiment definitions
        self.expts = [
            ("optic_flow", self.expt_optic_flow, 
             getattr(self, 'optic_flow_control_point', None),
             getattr(self, 'optic_flow_num_trials', None)),

            ("landmark", self.expt_landmark, 
             getattr(self, 'landmark_control_point', None),
             getattr(self, 'landmark_num_trials', None))
        ]

        # More config
        self.model_save_path = os.path.join('/Users/devenshidfar/Desktop/Masters/NRSC_510B/',
                                            'cebra_control_recal/models')
        self.save_models = 1
        self.save_anim = 1
        self.load_npy = 0
        self.rm_outliers = True
        self.vel_threshold = 5  # degrees per second
        self.bin_sizes = [1]
        self.max_num_reruns = max_num_reruns

        # For final data collection
        self.all_neural_data = []
        self.all_embeddings_3d = []
        self.all_principal_curves_3d = []
        self.all_curve_params_3d = []
        self.all_binned_hipp_angle = []
        self.all_binned_true_angle = []
        self.all_binned_est_gain = []
        self.all_binned_high_vel = []
        self.all_decoded_angles = []
        self.all_filtered_decoded_angles_unwrap = []
        self.all_decode_H = []
        self.all_session_idx = []
        self.all_rat = []
        self.all_day = []
        self.all_epoch = []
        self.all_num_skipped_clusters = []
        self.all_num_used_clusters = []
        self.all_avg_skipped_cluster_isolation_quality = []
        self.all_avg_used_cluster_isolation_quality = []
        self.all_mean_distance_to_principal_curve = []
        self.all_mean_angle_difference = []
        self.all_shuffled_mean_angle_difference = []
        self.all_SI_score_hipp = []
        self.all_SI_score_true = []
        self.all_mse_decode_vs_true = []
        self.all_mean_H_difference = []
        self.all_std_H_difference = []

    def run_analysis(self):
        """
        Main analysis pipeline. 
        Loops over expts (optic_flow, landmark), 
        loads sessions, runs embeddings, decodes, etc.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats
        from matplotlib.backends.backend_pdf import PdfPages
        from scipy.signal import savgol_filter
        from scipy.io import savemat


        for expt_name, expt, control_point, num_trials in self.expts:
            control_count = 0
            print(f"Control point: {control_point} and num_trials: {num_trials}")
            if control_point is None or num_trials is None:
                # If session_choose=True, skip the logic of skip/stop
                control_point = 0
                num_trials = len(expt)

            # For each bin size
            for bin_size in self.bin_sizes:
                print(f"\n[INFO] Processing expt: {expt_name}, bin_size: {bin_size}s")
                for session_idx, session in enumerate(expt):
                    try:
                        session_base_path = os.path.join(
                        '/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results',
                        self.save_folder,
                        expt_name,
                        f'rat_{session.rat}',
                        f'session_{session_idx}'
                        )   
                        os.makedirs(session_base_path, exist_ok=True)

                        SI_plots_path = os.path.join(session_base_path, 'SI_Plots')
                        anim_save_file = os.path.join(session_base_path, '3d_animations')
                        spectrogram_path = os.path.join(session_base_path, 'spatial_spectrograms')
                        param_plot_path = os.path.join(session_base_path, 'Param_Plots')

                        paths_to_create = [SI_plots_path, anim_save_file, spectrogram_path, param_plot_path]
                        for path in paths_to_create:
                            os.makedirs(path, exist_ok=True)

                        control_count += 1
                        # Skip control sessions
                        if control_count <= control_point:
                            print(f"Skipping session {session_idx + 1} (control count <= control_point).")
                            continue
                        elif control_count > (control_point + num_trials):
                            print("[INFO] Reached desired number of trials. Exiting session loop.")
                            break

                        print(f"\n[INFO] Processing session {session_idx + 1}/{len(expt)} in {expt_name} experiments")
                        print(f"Rat: {session.rat}, Day: {session.day}, Epoch: {session.epoch}")

                        ros_data = session.rosdata
                        start_time = ros_data.startTs
                        end_time = ros_data.stopTs

                        enc_times = np.array(ros_data.encTimes - start_time) / 1e6
                        vel = np.array(ros_data.vel)
                        valid_idx = np.isfinite(enc_times) & np.isfinite(vel)
                        enc_times = enc_times[valid_idx]
                        vel = vel[valid_idx]
                        high_vel_idx = vel > self.vel_threshold
                        if np.sum(high_vel_idx) == 0:
                            print("[WARNING] No data points above velocity threshold. Skipping session.")
                            continue

                        enc_times_high_vel = enc_times[high_vel_idx]
                        high_vel_filtered = vel[high_vel_idx]
                        est_gain_filtered = np.array(ros_data.estGain)[valid_idx][high_vel_idx]
                        hipp_angle_filtered = np.array(ros_data.hippAngle)[valid_idx][high_vel_idx]
                        true_angle_filtered = np.array(ros_data.encAngle)[valid_idx][high_vel_idx]
                        rel_angle_filtered = np.array(ros_data.relAngle)[valid_idx][high_vel_idx]

                        bins = np.arange(enc_times_high_vel[0], enc_times_high_vel[-1] + bin_size, bin_size)
                        if len(bins) < 2:
                            print("[WARNING] Not enough bins after filtering for high velocity. Skipping session.")
                            continue
                        
                        try:
                            binned_est_gain, _, _ = stats.binned_statistic(
                                enc_times_high_vel, est_gain_filtered, statistic='mean', bins=bins
                            )
                            binned_hipp_angle, _, _ = stats.binned_statistic(
                                enc_times_high_vel, hipp_angle_filtered, statistic='mean', bins=bins
                            )
                            binned_true_angle, _, _ = stats.binned_statistic(
                                enc_times_high_vel, true_angle_filtered, statistic='mean', bins=bins
                            )
                            binned_high_vel, _, _ = stats.binned_statistic(
                                enc_times_high_vel, high_vel_filtered, statistic='mean', bins=bins
                            )
                            binned_rel_angle, _, _ = stats.binned_statistic(
                                enc_times_high_vel, rel_angle_filtered, statistic='mean', bins=bins
                            )
                        except ValueError as e:
                            # Catch "Bin edges must be unique" 
                            print(f"[WARNING] Binning failed for session {session_idx}. Exception: {e}")
                            # Skip this session and continue with the next
                            continue

                        valid_bins = (
                            ~np.isnan(binned_hipp_angle) &
                            ~np.isnan(binned_true_angle) &
                            ~np.isnan(binned_est_gain) &
                            ~np.isnan(binned_high_vel)
                        )
                        if not np.all(valid_bins):
                            binned_hipp_angle = binned_hipp_angle[valid_bins]
                            binned_true_angle = binned_true_angle[valid_bins]
                            binned_est_gain = binned_est_gain[valid_bins]
                            binned_high_vel = binned_high_vel[valid_bins]
                            binned_rel_angle = binned_rel_angle[valid_bins]
                            bins = bins[:-1][valid_bins]

                        # Filter spike times
                        all_spikes = []
                        skipped_clusters = 0
                        used_clusters = 0
                        used_cluster_iq_list = []
                        skipped_cluster_iq_list = []

                        for cluster in session.clust:
                            if cluster.isolationQuality > 4:
                                skipped_clusters += 1
                                skipped_cluster_iq_list.append(cluster.isolationQuality)
                                continue
                            else:
                                used_clusters += 1
                                used_cluster_iq_list.append(cluster.isolationQuality)

                            spike_times_sec = (cluster.ts - start_time) / 1e6
                            vel_at_spikes = cluster.vel
                            include_spikes = vel_at_spikes > self.vel_threshold
                            spike_times_sec_high_vel = spike_times_sec[include_spikes]
                            if len(spike_times_sec_high_vel) == 0:
                                continue
                            
                            try:
                                binned_spikes, _, _ = stats.binned_statistic(
                                    spike_times_sec_high_vel,
                                    np.ones_like(spike_times_sec_high_vel),
                                    statistic='sum',
                                    bins=bins
                                )
                                all_spikes.append(binned_spikes)
                            except ValueError as e:
                                # Catch "Bin edges must be unique"
                                print(f"[WARNING] Binning failed for session {session_idx}. Exception: {e}")
                                # Skip this session and continue with the next
                                continue

                        # Stats for cluster IQ
                        used_cluster_iq = np.asarray(used_cluster_iq_list)
                        skipped_cluster_iq = np.asarray(skipped_cluster_iq_list)
                        num_skipped_cluster = len(skipped_cluster_iq)
                        num_used_cluster = len(used_cluster_iq)
                        avg_skipped_cluster_iq = np.mean(skipped_cluster_iq) if len(skipped_cluster_iq) else np.nan
                        avg_used_cluster_iq = np.mean(used_cluster_iq) if len(used_cluster_iq) else np.nan

                        if not all_spikes:
                            print("[WARNING] No valid spike data after filtering. Skipping session.")
                            continue

                        # Build neural data
                        neural_data = np.array(all_spikes).T
                        num_bins_neural = neural_data.shape[0]
                        num_bins_behavior = len(binned_est_gain)
                        if num_bins_neural != num_bins_behavior:
                            min_bins = min(num_bins_neural, num_bins_behavior)
                            neural_data = neural_data[:min_bins, :]
                            binned_est_gain = binned_est_gain[:min_bins]
                            binned_hipp_angle = binned_hipp_angle[:min_bins]
                            binned_true_angle = binned_true_angle[:min_bins]
                            binned_high_vel = binned_high_vel[:min_bins]
                            bins = bins[:min_bins]

                        # For reruns if SI < threshold
                        pdf_filename = os.path.join(session_base_path, f'session_{session_idx}.pdf')
                        os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
                        pdf = PdfPages(pdf_filename)

                        temperature_list = [1]
                        best_embeddings_3d = None
                        best_SI_score_hipp = -999

                        # Attempt multiple runs if SI < some threshold
                        for temp in temperature_list:
                            rerun_count = 0
                            while rerun_count < self.max_num_reruns:
                                embeddings_high_dim = CEBRAUtils.apply_cebra(
                                    neural_data=neural_data,
                                    output_dimension=3,
                                    temperature=temp
                                )
                                embeddings_3d = embeddings_high_dim.copy()

                                # Build mask for NaNs
                                nan_mask_3d = (
                                    ~np.isnan(embeddings_3d).any(axis=1) &
                                    ~np.isnan(binned_hipp_angle) &
                                    ~np.isnan(binned_true_angle) &
                                    ~np.isnan(binned_est_gain) &
                                    ~np.isnan(binned_high_vel)
                                )
                                embeddings_3d = embeddings_3d[nan_mask_3d, :]
                                binned_hipp_angle_temp = binned_hipp_angle[nan_mask_3d]
                                binned_true_angle_temp = binned_true_angle[nan_mask_3d]
                                binned_est_gain_temp = binned_est_gain[nan_mask_3d]
                                binned_high_vel_temp = binned_high_vel[nan_mask_3d]

                                # Outlier removal
                                if self.rm_outliers:
                                    inlier_indices_3d = CEBRAUtils.nt_TDA(embeddings_3d)
                                    embeddings_3d = embeddings_3d[inlier_indices_3d, :]
                                    binned_hipp_angle_temp = binned_hipp_angle_temp[inlier_indices_3d]
                                    binned_true_angle_temp = binned_true_angle_temp[inlier_indices_3d]
                                    binned_est_gain_temp = binned_est_gain_temp[inlier_indices_3d]
                                    binned_high_vel_temp = binned_high_vel_temp[inlier_indices_3d]

                                if(embeddings_3d.shape[0] < 1200):
                                    print(f"length of embeddings is: "
                                          "{embeddings_3d.shape[0]}, skipping session {session_idx}")

                                # Convert angles to radians
                                binned_true_angle_rad = np.deg2rad(binned_true_angle_temp)
                                binned_hipp_angle_rad = np.deg2rad(binned_hipp_angle_temp)

                                binned_true_angle_rad = (binned_true_angle_rad 
                                                        % (2 * np.pi))
                                binned_hipp_angle_rad = (binned_hipp_angle_rad 
                                                        % (2 * np.pi))

                                binned_true_angle_unwrap = binned_true_angle_rad
                                binned_hipp_angle_unwrap = binned_hipp_angle_rad

                                # Compute SI on 3D embeddings with hippocampal angle
                                SI_params = {
                                    'n_bins': 10,
                                    'n_neighbors': 15,
                                    'discrete_label': False,
                                    'num_shuffles': 10,
                                    'verbose': False,
                                }
                                

                                SI_score_hipp = CEBRAUtils.compute_SI_and_plot(
                                    embeddings=embeddings_3d,
                                    behav_var=binned_hipp_angle_rad,
                                    params=SI_params,
                                    behav_var_name='Hipp_Angle',
                                    save_dir=SI_plots_path,
                                    session_idx=session_idx,
                                    dimensions_3=True,
                                    pdf=pdf,
                                    num_used_clusters=num_used_cluster
                                )
                                if SI_score_hipp > best_SI_score_hipp:
                                    best_SI_score_hipp = SI_score_hipp
                                    best_embeddings_3d = embeddings_3d.copy()

                                if SI_score_hipp >= 0.8:
                                    break
                                elif SI_score_hipp < 0.8 and self.max_num_reruns > 1:
                                    print(f"[INFO] SI_score_hipp is {SI_score_hipp}. Retrying embedding.")
                                    rerun_count += 1

                            # If we tried max_num_reruns times, keep the best anyway
                            if best_embeddings_3d is not None:
                                embeddings_3d = best_embeddings_3d
                            else:
                                # fallback
                                embeddings_3d = embeddings_high_dim

                            # Compute SI with true angle
                            SI_score_true = CEBRAUtils.compute_SI_and_plot(
                                embeddings=embeddings_3d,
                                behav_var=np.deg2rad(binned_true_angle_temp),
                                params=SI_params,
                                behav_var_name='True_Angle',
                                save_dir=os.path.join(
                                    os.path.dirname(pdf_filename),
                                    'SI_Plots'
                                ),
                                session_idx=session_idx,
                                dimensions_3=True,
                                pdf=pdf,
                                num_used_clusters=num_used_cluster
                            )

                            # Fit principal curve (SPUD)
                            fit_params = {
                                'dalpha': 0.005,
                                'knot_order': 'nearest',
                                'penalty_type': 'curvature',
                                'nKnots': 20,
                                'curvature_coeff': 5,
                                'len_coeff': 2,
                                'density_coeff': 2,
                                'delta': 0.1
                            }

                            principal_curve_3d, principal_curve_3d_pre, curve_params_3d = CEBRAUtils.fit_spud_to_cebra(
                                embeddings=embeddings_3d,
                                ref_angle=binned_true_angle_unwrap,
                                session_idx=session_idx,
                                session=session,
                                results_save_path=os.path.join(os.path.dirname(pdf_filename)),
                                fit_params=fit_params,
                                dimension_3d=1,
                                verbose=False
                            )

                            if principal_curve_3d is None:
                                print("[INFO] Detected knots were too close. "
                                      "Skipping the entire session and writing NaNs.")

                                # Here is an example block showing how you might store NaNs:
                                nan_array = np.full_like(binned_true_angle_unwrap, np.nan)
                                self.all_neural_data.append(np.nan)
                                self.all_embeddings_3d.append(np.nan)
                                self.all_principal_curves_3d.append(np.nan)
                                self.all_curve_params_3d.append(np.nan)
                                self.all_binned_hipp_angle.append(nan_array)
                                self.all_binned_true_angle.append(nan_array)
                                self.all_binned_est_gain.append(nan_array)
                                self.all_binned_high_vel.append(nan_array)
                                self.all_decoded_angles.append(nan_array)
                                self.all_filtered_decoded_angles_unwrap.append(nan_array)
                                self.all_decode_H.append(nan_array)
                                self.all_session_idx.append(session_idx)
                                self.all_rat.append(session.rat)
                                self.all_day.append(session.day)
                                self.all_epoch.append(session.epoch)
                                self.all_num_skipped_clusters.append(np.nan)
                                self.all_num_used_clusters.append(np.nan)
                                self.all_avg_skipped_cluster_isolation_quality.append(np.nan)
                                self.all_avg_used_cluster_isolation_quality.append(np.nan)
                                self.all_mean_distance_to_principal_curve.append(np.nan)
                                self.all_mean_angle_difference.append(np.nan)
                                self.all_shuffled_mean_angle_difference.append(np.nan)
                                self.all_SI_score_hipp.append(np.nan)
                                self.all_SI_score_true.append(np.nan)
                                self.all_mse_decode_vs_true.append(np.nan)
                                self.all_mean_H_difference.append(np.nan)
                                self.all_std_H_difference.append(np.nan)

                                # Continue to the next session if principal curve is None
                                continue

                            # Distance to principal curve
                            mean_dist_to_princ = CEBRAUtils.mean_dist_to_spline(embeddings_3d, principal_curve_3d)

                            # Decode angles
                            decoded_angles, mse_decode_vs_true = CEBRAUtils.decode_hipp_angle_spline(
                                embeddings=embeddings_3d,
                                principal_curve=principal_curve_3d,
                                tt=curve_params_3d,
                                behav_angles=binned_hipp_angle_rad,
                                true_angles=binned_true_angle_rad
                            )
                            # Shuffled
                            shuffled_hipp = np.random.permutation(binned_hipp_angle_rad)
                            shuffled_decoded_angles, shuffled_mse_decode_vs_true = CEBRAUtils.decode_hipp_angle_spline(
                                embeddings=embeddings_3d,
                                principal_curve=principal_curve_3d,
                                tt=curve_params_3d,
                                behav_angles=shuffled_hipp,
                                true_angles=binned_true_angle_rad
                            )

                            decoded_angles_unwrap = decoded_angles + binned_true_angle_unwrap[3]
                            shuffled_decoded_angles_unwrap = shuffled_decoded_angles + binned_true_angle_unwrap[3]
                            decoded_angles = (decoded_angles_unwrap) % (2 * np.pi)
                            shuffled_decoded_angles = (shuffled_decoded_angles_unwrap) % (2 * np.pi)
                            angle_diff = (decoded_angles - binned_hipp_angle_rad) % (2 * np.pi)
                            shuffled_angle_diff = (shuffled_decoded_angles - binned_hipp_angle_rad) % (2 * np.pi)
                            mean_angle_diff = np.mean(angle_diff)
                            shuffled_mean_angle_diff = np.mean(shuffled_angle_diff)

                            # Low-pass filter
                            filtered_decoded_angles_unwrap = CEBRAUtils.low_pass_filter(
                                angles=decoded_angles_unwrap, cutoff_frequency=0.2, filter_order=3, fs=1
                            )
                            filtered_decoded_angles_unwrap = savgol_filter(filtered_decoded_angles_unwrap, window_length=30, polyorder=2)

                            derivative_decoded_angle = CEBRAUtils.window_smooth(
                                data=filtered_decoded_angles_unwrap, window_size=60
                            )
                            derivative_true_angle = CEBRAUtils.window_smooth(
                                data=binned_true_angle_unwrap, window_size=60
                            )
                            derivative_hipp_angle = CEBRAUtils.window_smooth(
                                data=binned_hipp_angle_unwrap, window_size=60
                            )

                            # Plot the filter result
                            plt.figure(figsize=(15, 6))
                            plt.plot(decoded_angles_unwrap, label='Original Decoded', alpha=0.5)
                            plt.plot(shuffled_decoded_angles_unwrap, label='Shuffled Decoded', alpha=0.5)
                            plt.plot(filtered_decoded_angles_unwrap, label='Filtered Decoded', linewidth=2)
                            plt.xlabel('Time (s)')
                            plt.ylabel('Decoded Angle (rad)')
                            plt.title('Low-Pass Filter Applied to Decoded Angles')
                            plt.legend()
                            plt.grid(True)
                            pdf.savefig()
                            plt.close()

                            # Save side-by-side plots in PDF
                            os.makedirs(param_plot_path, exist_ok=True)
                            CEBRAUtils.plot_decoded_var_and_true(
                                decoded_var=filtered_decoded_angles_unwrap,
                                behav_var=binned_hipp_angle_unwrap,
                                save_path=param_plot_path,
                                session_idx=session_idx,
                                behav_var_name='Hipp',
                                pdf=pdf,
                                legend_labels=['Filtered Decoded', 'Hipp Angle']
                            )

            

                            # Static 3D plot (example)
                            from mpl_toolkits.mplot3d import Axes3D
                            fig_3d = plt.figure(figsize=(10, 8))
                            ax3d = fig_3d.add_subplot(111, projection='3d')
                            scatter = ax3d.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                                                c=binned_hipp_angle_rad, cmap='viridis', s=5)
                            if principal_curve_3d is not None:
                                ax3d.plot(principal_curve_3d[:, 0], principal_curve_3d[:, 1], principal_curve_3d[:, 2],
                                        color='red', linewidth=2)
                            fig_3d.colorbar(scatter, label='Hipp Angle (rad)')
                            ax3d.set_title(f"3D Embeddings (Session {session_idx})")

                            # 3D plot
                            # plot_file_path = os.path.join(anim_save_file, f"3d_plot_session_{session_idx}.png")

                            fig_3d.savefig(f"{anim_save_file}/3d_plot_session_{session_idx}.png", format='png', dpi=300, bbox_inches='tight')
                            pdf.savefig(fig_3d)
                            plt.close(fig_3d)

                            # Now compute decode_H
                            decode_H = derivative_decoded_angle / derivative_true_angle
                            decode_H_nan_count = np.isnan(decode_H).sum()
                            decode_H = np.clip(decode_H, -2, 2)

                            min_len = min(len(binned_est_gain_temp), len(decode_H))
                            mean_H_diff = np.mean(np.abs(binned_est_gain_temp[:min_len] - decode_H[:min_len]))
                            std_H_diff = np.std(binned_est_gain_temp[:min_len] - decode_H[:min_len])

                            # Behav vars over lapss
                            lap_number, sorted_decode_H, sorted_lap_number = CEBRAUtils.get_var_over_lap(
                                var=decode_H, true_angle=binned_true_angle_unwrap
                            )
                            _, sorted_H_est, _ = CEBRAUtils.get_var_over_lap(
                                var=binned_est_gain_temp, true_angle=binned_true_angle_unwrap
                            )
                            _, sorted_vel, _ = CEBRAUtils.get_var_over_lap(
                                var=binned_high_vel_temp, true_angle=binned_true_angle_unwrap
                            )

                            # Hipp frame
                            hipp_lap_number, hipp_decode_H, hipp_sorted_lap_number = CEBRAUtils.get_var_over_lap(
                                var=decode_H, true_angle=binned_hipp_angle_unwrap
                            )
                            # _, sorted_rel_angle, _ = CEBRAUtils.get_var_over_lap(
                            #     var=np.deg2rad(rel_angle_filtered[valid_bins][nan_mask_3d]), 
                            #     true_angle=binned_true_angle_unwrap
                            # ) if len(rel_angle_filtered) else (None, None, None)

                            # Spatial spectrogram
                            
                            os.makedirs(spectrogram_path, exist_ok=True)
                            CEBRAUtils.plot_spatial_spectrogram(
                                H=sorted_decode_H, lap_numbers=sorted_lap_number,
                                behav_var_name="True_Frame",
                                save_path=spectrogram_path,
                            )
                            CEBRAUtils.plot_spatial_spectrogram(
                                H=hipp_decode_H, lap_numbers=hipp_sorted_lap_number,
                                behav_var_name="Hipp_Frame",
                                save_path=spectrogram_path,
                            )

                            # Save final results
                            self.all_neural_data.append(neural_data)
                            self.all_embeddings_3d.append(embeddings_3d)
                            self.all_principal_curves_3d.append(principal_curve_3d)
                            self.all_curve_params_3d.append(curve_params_3d)
                            self.all_binned_hipp_angle.append(binned_hipp_angle_unwrap)
                            self.all_binned_true_angle.append(binned_true_angle_unwrap)
                            self.all_binned_est_gain.append(binned_est_gain_temp)
                            self.all_binned_high_vel.append(binned_high_vel_temp)
                            self.all_decoded_angles.append(decoded_angles_unwrap)
                            self.all_filtered_decoded_angles_unwrap.append(filtered_decoded_angles_unwrap)
                            self.all_decode_H.append(decode_H)
                            self.all_session_idx.append(session_idx)
                            self.all_rat.append(session.rat)
                            self.all_day.append(session.day)
                            self.all_epoch.append(session.epoch)
                            self.all_num_skipped_clusters.append(num_skipped_cluster)
                            self.all_num_used_clusters.append(num_used_cluster)
                            self.all_avg_skipped_cluster_isolation_quality.append(avg_skipped_cluster_iq)
                            self.all_avg_used_cluster_isolation_quality.append(avg_used_cluster_iq)
                            self.all_mean_distance_to_principal_curve.append(mean_dist_to_princ)
                            self.all_mean_angle_difference.append(mean_angle_diff)
                            self.all_shuffled_mean_angle_difference.append(shuffled_mean_angle_diff)
                            self.all_SI_score_hipp.append(best_SI_score_hipp)
                            self.all_SI_score_true.append(SI_score_true)
                            self.all_mse_decode_vs_true.append(mse_decode_vs_true)
                            self.all_mean_H_difference.append(mean_H_diff)
                            self.all_std_H_difference.append(std_H_diff)

                        pdf.close()

                        # After finishing all expts, compile data into a .mat file
                        data_dict = {
                            'neural_data': self.all_neural_data,
                            'embeddings_3d': self.all_embeddings_3d,
                            'principal_curves_3d': self.all_principal_curves_3d,
                            'curve_params_3d': self.all_curve_params_3d,
                            'binned_hipp_angle': self.all_binned_hipp_angle,
                            'binned_true_angle': self.all_binned_true_angle,
                            'binned_est_gain': self.all_binned_est_gain,
                            'binned_high_vel': self.all_binned_high_vel,
                            'decoded_angles': self.all_decoded_angles,
                            'filtered_decoded_angles_unwrap': self.all_filtered_decoded_angles_unwrap,
                            'decode_H': self.all_decode_H,
                            'session_idx': self.all_session_idx,
                            'rat': self.all_rat,
                            'day': self.all_day,
                            'epoch': self.all_epoch,
                            'num_skipped_clusters': self.all_num_skipped_clusters,
                            'num_used_clusters': self.all_num_used_clusters,
                            'avg_skipped_cluster_isolation_quality': self.all_avg_skipped_cluster_isolation_quality,
                            'avg_used_cluster_isolation_quality': self.all_avg_used_cluster_isolation_quality,
                            'mean_distance_to_principal_curve': self.all_mean_distance_to_principal_curve,
                            'mean_angle_difference': self.all_mean_angle_difference,
                            'shuffled_mean_angle_difference': self.all_shuffled_mean_angle_difference,
                            'SI_score_hipp': self.all_SI_score_hipp,
                            'SI_score_true': self.all_SI_score_true,
                            'mse_decode_vs_true': self.all_mse_decode_vs_true,
                            'mean_H_difference': self.all_mean_H_difference,
                            'std_H_difference': self.all_std_H_difference
                        }

                        base_path = f'/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results/'
                        mat_filename = os.path.join(base_path, f'{self.save_folder}_all_sessions_data.mat')
                        savemat(mat_filename, data_dict)
                        print(f"[INFO] Saved final data to {mat_filename}")
                    
                    except AttributeError as e:
                        print(f"[WARNING] Session {session_idx + 1} is invalid or missing. Skipping...")
                        print(f"Error: {e}")
                        continue

                    #end Expt loop

def main():
    """
    Entry point to run the entire analysis.
    """
    analysis = CEBRAAnalysis(session_choose=False, max_num_reruns=1)
    analysis.run_analysis()

if __name__ == "__main__":
    main()
