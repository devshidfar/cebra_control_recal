import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
from scipy.io import savemat
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import cebra
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objs as go
from plotly.offline import plot
from matplotlib.widgets import Slider
import mpld3
from ripser import ripser
from persim import plot_diagrams
from dataclasses import dataclass, asdict
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go
import plotly.io as pio

# CLASS: CEBRAUtils


@dataclass
class SessionData:
    expt_file: any  # Replace `any` with the actual type if known
    specifier: np.ndarray
    neural_data_fit: np.ndarray
    neural_data_low_vel: np.ndarray
    embeddings_low_vel: np.ndarray
    neural_data_full_trial: np.ndarray
    neural_data_land_off: np.ndarray
    neural_data_land_on: np.ndarray
    embeddings_3d: np.ndarray
    high_vel_mask: np.ndarray
    H0_value: float
    H1_value: float
    betti_0: int
    betti_1: int
    principal_curves_3d: np.ndarray
    curve_params_3d: any 
    binned_hipp_angle: np.ndarray
    binned_true_angle: np.ndarray
    binned_est_gain: np.ndarray
    binned_vel: np.ndarray
    decoded_angles: np.ndarray
    filtered_decoded_angles_unwrap: np.ndarray
    decode_H: np.ndarray
    lap_decode_H: np.ndarray
    lap_est_gain: np.ndarray
    lap_vel: np.ndarray
    lap_number: np.ndarray
    hipp_lap_number: np.ndarray
    session_idx: int
    rat: any  
    day: any  
    epoch: any  
    num_skipped_clusters: int
    num_used_clusters: int
    avg_skipped_cluster_iq: float
    avg_used_cluster_iq: float
    mean_dist_to_spline: float
    dist_to_spline: np.ndarray
    mean_angle_diff: float
    shuffled_mean_angle_diff: float
    SI_score_hipp: float
    SI_score_true: float
    mean_H_difference: float
    std_H_difference: float

class CEBRAUtils:
    """
    Static utility methods for applying CEBRA, computing metrics, plotting, etc.
    These were originally from 'CEBRA_Utils.py'.
    """

    @staticmethod
    def load_optic_flow_data(path_optic_flow):

            data_optic_flow = scipy.io.loadmat(
                path_optic_flow, 
                squeeze_me=True, 
                struct_as_record=False
            )
            expt_optic_flow = data_optic_flow['expt']
            print(f"Optic Flow expt shape: {expt_optic_flow.shape}")
            return expt_optic_flow

    @staticmethod
    def load_landmark_data(path_landmark):
        data_landmark = scipy.io.loadmat(
            path_landmark, 
            squeeze_me=True, 
            struct_as_record=False
        )
        expt_landmark = data_landmark['expt']
        print(f"Landmark expt shape: {expt_landmark.shape}")
        return expt_landmark

    

    @staticmethod
    def apply_cebra(neural_data_fit=None, neural_data_embeddings=None, neural_data_low_vel=None, output_dimension=7, temperature=1):
        """
        Apply the CEBRA model to 'neural_data' and return the embeddings.
        """
        model = cebra.CEBRA(
            output_dimension=output_dimension,
            max_iterations=1000,
            batch_size=512,
            temperature=temperature
        )
        model.fit(neural_data_fit)
        embeddings = model.transform(neural_data_embeddings)
        if(neural_data_low_vel.shape[0] >= 1):
            low_vel_embeddings = model.transform(neural_data_low_vel)
        else:
            low_vel_embeddings = None
        return embeddings, low_vel_embeddings
    
    @staticmethod
    def linear_interpolate_nans_2d(array_2d):
        """
        Replace NaNs with linearly interpolated values (per column).
        array_2d has shape (N, dim).
        """
        x = np.arange(len(array_2d))
        for dim in range(array_2d.shape[1]):
            col = array_2d[:, dim]
            # Identify where col is non-NaN
            good = ~np.isnan(col)
            if np.all(~good):
                # If the entire column is NaN, we can't interpolate. 
                continue
            # Interpolate over the NaNs (including boundaries)
            array_2d[~good, dim] = np.interp(x[~good], x[good], col[good])
        return array_2d
    
    @staticmethod
    def linear_interpolate_nans_1d(array_1d):
        """
        Replaces NaN entries in a 1D NumPy array by linear interpolation.
        """
        x = np.arange(len(array_1d))
        good = ~np.isnan(array_1d)
        if np.all(~good):
            # If the entire array is NaN, do nothing or fill as needed
            return array_1d
        array_1d[~good] = np.interp(x[~good], x[good], array_1d[good])
        return array_1d
        
    @staticmethod
    def run_persistent_homology(
        embeddings, 
        session_idx=None, 
        session=None, 
        results_save_path=None, 
        dimension=3
    ):
        """
        Runs persistent homology (via 'ripser') on the given 'embeddings',
        plots & saves the corresponding persistence diagrams, and optionally
        stores the raw results. Dimension typically = 2 or 3, but can be higher.

        Parameters:
        - embeddings (np.ndarray): shape (N, dim), the points for PH analysis
        - session_idx (int, optional): session index, for labeling/saving
        - session (object, optional): session metadata (rat, day, epoch)
        - results_save_path (str, optional): Where to save outputs
        - dimension (int, optional): Dimension of embeddings (2, 3, etc.)
        """

        # Create a folder for storing PH results
        ph_dir = os.path.join(results_save_path, "persistent_homology", f"session_{session_idx}", f"dimension_{dimension}")
        os.makedirs(ph_dir, exist_ok=True)

        # Run ripser (maxdim=1 => compute H0, H1)
        print(f"[INFO] Computing persistent homology for session {session_idx}, dimension {dimension} ...")
        ph_result = ripser(embeddings, maxdim=1)  
        diagrams = ph_result['dgms']   # diagram[0] = H0, diagram[1] = H1

        #return H0 and H1 values

        H0_values = diagrams[0]
        H1_values = diagrams[1]

        betti_0, betti_1 = CEBRAUtils.compute_betti_numbers(H0_values, H1_values, lifespan_fraction=0.5)

        # Save raw results
        np.savez(os.path.join(ph_dir, "ripser_result.npz"), H0=diagrams[0], H1=diagrams[1])
        print(f"[INFO] Saved raw PH results (H0 & H1) to {ph_dir}")



        # Plot and save the persistence diagram
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # H0 plot
        if diagrams[0].size == 0:
            print("[DEBUG] No H0 features found in the persistence diagram. Skipping H0 Barcode plot.")
        else:
            plot_diagrams(diagrams[0], show=False, ax=ax[0], title="H0 Barcode")

        # H1 plot
        if diagrams[1].size == 0:
            print("[DEBUG] No H1 features found in the persistence diagram. Skipping H1 Barcode plot.")
        else:
            plot_diagrams(diagrams[1], show=False, ax=ax[1], title="H1 Barcode")

        if session:
            fig.suptitle(f"Persistent Homology\nSession={session_idx}, Rat={session.rat}, Day={session.day}, Epoch={session.epoch}")
        else:
            fig.suptitle(f"Persistent Homology - Session {session_idx}")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_save_path = os.path.join(ph_dir, "barcode_diagrams.png")
        plt.savefig(fig_save_path, dpi=300)
        plt.close(fig)

        print(f"[INFO] Saved persistence diagrams to: {fig_save_path}")
        
        # If you want the standard "persistence diagram" style, you can also do:
        #   plot_diagrams(diagrams, show=True)
        # or store them in additional files.`

        return H0_values, H1_values, betti_0, betti_1
    
    @staticmethod
    def compute_betti_numbers(H0_array, H1_array, lifespan_fraction=0.5):
        def count_features(diagram, frac):
            if diagram.size == 0:
                return 0
            lifespans = diagram[:, 1] - diagram[:, 0]
            lifespans = lifespans[np.isfinite(lifespans)]
            if len(lifespans) == 0:
                return 0
            max_lifespan = np.max(lifespans)
            threshold = max_lifespan * frac
            return int(np.sum(lifespans > threshold))

        betti_0 = count_features(H0_array, lifespan_fraction)
        betti_1 = count_features(H1_array, lifespan_fraction)
        return betti_0, betti_1

    
    @staticmethod
    def decode_hipp_angle_spline(
            embeddings=None,
            principal_curve=None,
            tt=None,
            true_angles=None,
    ):
        """
        Calculate the hippocampal angle decoded from the spline.
        """
        from manifold_fit_and_decode_fns_custom import decode_from_passed_fit
            
        decoded_angles = decode_from_passed_fit(
            embeddings, 
            tt[:-1], 
            principal_curve[:-1], 
            true_angles
        )

        return decoded_angles

    @staticmethod
    def derivative_and_mv_avg(data=None, window_size=3):
        """
        Compute finite differences and then the moving average
        of those differences, essentially a smoothing operation.
        """
        diffs = np.diff(data)
        # Add the same value as the second element to keep the temporality as before
        diffs = np.insert(diffs, 0, data[0]) 
        kernel = np.ones(window_size) / window_size
        avg_diffs = np.convolve(diffs, kernel, mode='same')
        # Output length = Input length - (Kernel Length - 1)
        return avg_diffs

    @staticmethod
    def nt_TDA_mask(data, pct_distance=1, pct_neighbors=50, pct_dist=80, verbose=False):
        """
        Detect outliers in 'data' using the TDA-inspired method,
        **but return a boolean mask** of the same length as 'data', 
        indicating which rows are inliers (True) vs. outliers (False).
        """
        from scipy.spatial import distance_matrix
        from sklearn.neighbors import NearestNeighbors

        # data.shape = (N, dim)
        N = data.shape[0]
        if(verbose == True):
            print(f"[DEBUG] Running nt_TDA_mask on data of shape {data.shape}")
            print(f"[DEBUG] Using params: pct_distance={pct_distance}, pct_neighbors={pct_neighbors}, pct_dist={pct_dist}")


        distances = distance_matrix(data, data)
        if(verbose == True):
            print(f"[DEBUG] distance_matrix shape: {distances.shape} (should be (N, N))")

        # Fill diagonal so points don't count themselves
        np.fill_diagonal(distances, 10)

        # Neighborhood radius for each point
        neighborhood_radius = np.percentile(distances, pct_distance, axis=0)
        if(verbose == True): 
            print(f"[DEBUG] neighborhood_radius: min={neighborhood_radius.min():.3f}, "
            f"max={neighborhood_radius.max():.3f}, median={np.median(neighborhood_radius):.3f}")

        neighbor_counts = np.sum(distances <= neighborhood_radius[:, None], axis=1)
        threshold_neighbors = np.percentile(neighbor_counts, pct_neighbors)
        if(verbose == True):
            print(f"[DEBUG] threshold_neighbors (pct_neighbors={pct_neighbors}) is {threshold_neighbors:.3f}")

        outlier_indices_1 = np.where(neighbor_counts < threshold_neighbors)[0]
        if(verbose == True):
            print(f"[DEBUG] outlier_indices_1 has length {len(outlier_indices_1)}")
        # "Far out" outliers using mean min-distance
        neighbgraph = NearestNeighbors(n_neighbors=5).fit(distances)
        dists, _ = neighbgraph.kneighbors(distances)
        min_distance_to_any_point = np.mean(dists, axis=1)
        if(verbose == True):
            print(f"[DEBUG] min_distance_to_any_point: min={min_distance_to_any_point.min():.3f}, "
            f"max={min_distance_to_any_point.max():.3f}, median={np.median(min_distance_to_any_point):.3f}")

        distance_threshold = np.percentile(min_distance_to_any_point, pct_dist)
        if(verbose == True):
            print(f"[DEBUG] distance_threshold (pct_dist={pct_dist}) is {distance_threshold:.3f}")

        outlier_indices_2 = np.where(min_distance_to_any_point > distance_threshold)[0]
        if(verbose == True):
            print(f"[DEBUG] outlier_indices_2 has length {len(outlier_indices_2)}")

        all_outliers = np.unique(np.concatenate([outlier_indices_1, outlier_indices_2]))
        if(verbose == True):
            print(f"[DEBUG] Total unique outliers: {len(all_outliers)}")
        # Construct boolean mask: True = inlier, False = outlier

        
    
        mask = np.ones(N, dtype=bool)
        mask[all_outliers] = False
        inlier_count = np.sum(mask)
        if(verbose == True):
            print(f"[DEBUG] Number of inliers: {inlier_count}, out of N={N}")
            print(f"[DEBUG] Outlier ratio = {1.0 - inlier_count / N:.3f}")
        


        return mask

    # From structure_index.py
    # @staticmethod
    # def nt_TDA_mask(data):
    #     D = pairwise_distances(data)
    #     np.fill_diagonal(D, np.nan)
    #     nn_dist = np.sum(D < np.nanpercentile(D,1), axis=1) - 1
    #     print(f"the number of nearest bottom 1 percent {nn_dist}")
    #     noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    #     print(f"the number of outliers {noiseIdx}")
    #     return noiseIdx

    @staticmethod
    def plot_embeddings_static(
                            embeddings = None,
                            embeddings_low_vel = None,
                            principal_curve = None,
                            behav_var_name='Low Vel',
                            behav_var=None,
                            session_idx=None,
                            save_path=None
                            ):

        # Static 3D plot of embeddings
        # Create figure
        fig_3d = plt.figure(figsize=(10, 8))
        ax3d = fig_3d.add_subplot(111, projection='3d')

        # Plot embeddings_3d with color mapping
        scatter = ax3d.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                                c=behav_var, cmap='jet', s=5, alpha=0.7)

        # Overlay low-velocity embeddings in pink
        ax3d.scatter(embeddings_low_vel[:, 0], embeddings_low_vel[:, 1], embeddings_low_vel[:, 2],
                    color='magenta', s=10, label=behav_var_name, alpha=0.8)

        # Plot principal curve if available
        if principal_curve is not None:
            ax3d.plot(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2],
                    color='red', linewidth=2, label="Principal Curve")

        # Add colorbar
        fig_3d.colorbar(scatter, label='Hipp Angle (rad)')

        # Labels & legend
        ax3d.set_title(f"3D Embeddings (Session {session_idx})")
        ax3d.legend()

        # Show the plot
        plt.show()

        plt.savefig(f"{save_path}/3D_embeddings_session{session_idx}",dpi=300,bbox_inches='tight')
    
        return
    
    @staticmethod
    def plot_embeddings_interactive(
                            embeddings=None,
                            embeddings_low_vel=None,
                            principal_curve=None,
                            behav_var_name='Low Vel',
                            behav_var=None,
                            session_idx=None,
                            save_html=True,
                            save_path=None,
                            html_filename="3D_Embeddings_Interactive.html"
                            ):
        """
        Interactive 3D plot of embeddings using Plotly.

        Parameters:
        -----------
        embeddings : np.array, shape (N, 3)
            3D embeddings, color-coded by `behav_var`.
        embeddings_low_vel : np.array, shape (M, 3)
            Low-velocity embeddings, plotted in pink.
        principal_curve : np.array, shape (K, 3), optional
            Principal curve to overlay.
        behav_var_name : str, optional
            Label for the low-velocity embeddings.
        behav_var : np.array, shape (N,)
            Values used for coloring the main embeddings.
        session_idx : int, optional
            Session index for labeling.
        save_html : bool, optional
            If True, saves the interactive figure as an HTML file.
        html_filename : str, optional
            Filename for the saved HTML file.
        """


        # Create a Plotly figure
        fig = go.Figure()

        # Scatter plot for embeddings (color-coded by behavioral variable)
        fig.add_trace(go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=behav_var,  # Color by behavior variable
                colorscale='jet',
                opacity=0.7,
                colorbar=dict(title="Hipp Angle (rad)")
            ),
            name="3D Embeddings"
        ))

        # Scatter plot for low-velocity embeddings (Pink)
        if embeddings_low_vel is not None:
            fig.add_trace(go.Scatter3d(
                x=embeddings_low_vel[:, 0],
                y=embeddings_low_vel[:, 1],
                z=embeddings_low_vel[:, 2],
                mode='markers',
                marker=dict(size=8, color='magenta', opacity=0.8),
                name=behav_var_name
            ))

        # Line plot for principal curve (if available)
        if principal_curve is not None:
            fig.add_trace(go.Scatter3d(
                x=principal_curve[:, 0],
                y=principal_curve[:, 1],
                z=principal_curve[:, 2],
                mode='lines',
                line=dict(color='red', width=4),
                name="Principal Curve"
            ))

        mean_embeddings = np.mean(embeddings, axis=0)
        mean_low_vel = np.mean(embeddings_low_vel, axis=0)

        text_annotation = f"""
        <b>Mean Embeddings:</b><br>
        X: {mean_embeddings[0]:.3f}, Y: {mean_embeddings[1]:.3f}, Z: {mean_embeddings[2]:.3f}<br>
        <b>Mean Low Vel:</b><br>
        X: {mean_low_vel[0]:.3f}, Y: {mean_low_vel[1]:.3f}, Z: {mean_low_vel[2]:.3f}
        """

        fig.add_annotation(
            text=text_annotation,
            showarrow=False,
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            bordercolor="black",
            borderwidth=2,
            bgcolor="white",
            opacity=0.8
        )

        # Set layout
        fig.update_layout(
            title=f"3D Interactive Embeddings (Session {session_idx})",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        save_file = os.path.join(save_path,html_filename)
        pio.write_html(fig,save_file)


        return fig  # Returning fig in case further customization is needed

    @staticmethod
    def fit_b_spline_to_embeddings(embeddings, ref_angle=None, session_idx=None, session=None, 
                               results_save_path=None, fit_params=None, dimension_3d=None, verbose=False):
        """
        Fits a B-spline curve to high-dimensional neural embeddings and outputs the spline knots.
        
        Parameters
        ----------
        embeddings : np.array, shape (N, d)
            High-dimensional data points (e.g., neural embeddings).
        ref_angle : array-like, optional
            Reference angle for alignment. For example, ref_angle[3] is used to shift the parameterization.
        session_idx : int, optional
            Session index (used for saving plots).
        session : any, optional
            Session identifier.
        results_save_path : str, optional
            Directory in which to save plots (if desired).
        fit_params : dict, optional
            Dictionary of parameters for fitting. Example:
                {
                    'num_knots': 10,         # Number of initial knots
                    'smoothing': 0,          # Smoothing factor (s) for splprep
                    'degree': 3,             # Degree of the B-spline (k)
                    'num_curve_points': 200, # Number of points at which to evaluate the final spline curve
                    'knot_order': 'pca'      # Method to order knots (here we use PCA)
                }
        dimension_3d : bool or int, optional
            If True (or 1) and the embeddings are 3D, a 3D scatter plot of the embeddings and initial knots is generated.
        verbose : bool, optional
            If True, prints debug statements.
            
        Returns
        -------
        curve_points : np.array, shape (M, d)
            Evaluated B-spline curve points.
        tck : tuple
            The tuple (t, c, k) representing the spline, where:
                t = knot vector,
                c = list of B-spline coefficients (control points for each dimension),
                k = degree of the spline.
        u_fine : np.array
            Parameter values corresponding to the curve_points.
        final_knots : np.array
            The ordered and refined initial knots from clustering.
        """
        # Set default fit parameters if none provided.
        if fit_params is None:
            fit_params = {
                'num_knots': 10,
                'smoothing': 0,
                'degree': 3,
                'num_curve_points': 200,
                'knot_order': 'pca'
            }
        
        # --- Step 1: Obtain Initial Knots via Clustering ---
        num_knots = fit_params.get('num_knots', 10)
        if verbose:
            print("Clustering embeddings to obtain initial knots...")
        # Using KMeans as a proxy for k-medoids clustering.
        kmeans = KMeans(n_clusters=num_knots, random_state=0).fit(embeddings)
        initial_knots = kmeans.cluster_centers_
        
        # --- Step 2: Order the Knots ---
        if verbose:
            print("Ordering knots using PCA...")
        # Project the initial knots onto the first principal component and sort.
        pca = PCA(n_components=1)
        proj = pca.fit_transform(initial_knots)
        order = np.argsort(proj[:, 0])
        ordered_knots = initial_knots[order]
        
        # --- Step 3: Remove Outlier Knots ---
        # Remove knots that are far from their neighbors (up to 5 removals).
        final_knots = ordered_knots.copy()
        knots_removed = 0
        while True:
            if final_knots.shape[0] < 2:
                break
            segments = np.diff(final_knots, axis=0)
            knot_dists = np.linalg.norm(segments, axis=1)
            max_dist = np.max(knot_dists)
            median_dist = np.median(knot_dists)
            if verbose:
                print(f"Max segment distance: {max_dist:.3f}, Median segment distance: {median_dist:.3f}")
            if max_dist > 1.5 * median_dist and knots_removed < 5:
                max_idx = np.argmax(knot_dists)
                final_knots = np.delete(final_knots, max_idx+1, axis=0)
                knots_removed += 1
                if verbose:
                    print(f"Removed knot at index {max_idx+1}. Total removed: {knots_removed}")
            else:
                break
        
        # --- Optional: Plot Initial/Final Knots (for 3D data) ---
        if dimension_3d and results_save_path is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c='gray', alpha=0.5, label='Embeddings')
            ax.scatter(final_knots[:, 0], final_knots[:, 1], final_knots[:, 2], c='red', label='Initial Knots')
            ax.set_title(f"Initial Knots for Session {session_idx}")
            ax.legend()
            plot_save_path = os.path.join(results_save_path, f"initial_knots_session_{session_idx}.png")
            plt.savefig(plot_save_path)
            plt.close()
        
        # --- Step 4: Fit a B-spline to the Embeddings ---
        if verbose:
            print("Fitting B-spline to embeddings using splprep...")
        # splprep expects the data as a list of 1D arrays (one per dimension).x
            print("Embeddings shape:", embeddings.shape)
            print("Embeddings dtype:", embeddings.dtype)

        dims = embeddings.shape[1]
        data = [embeddings[:, i] for i in range(dims)]
        print("Data shapes:", [d.shape for d in data])
        tck, u = splprep(data, s=fit_params.get('smoothing', 0), k=fit_params.get('degree', 3))
        # The tck tuple contains:
        #   tck[0]: the knot vector,
        #   tck[1]: the B-spline coefficients (control points for each dimension),
        #   tck[2]: the degree of the spline.
        
        # --- Step 5: Evaluate the Spline at Fine Parameter Values ---
        num_curve_points = fit_params.get('num_curve_points', 200)
        u_fine = np.linspace(0, 1, num_curve_points)
        curve_eval = splev(u_fine, tck)
        curve_points = np.stack(curve_eval, axis=-1)  # shape: (num_curve_points, dims)
        
        # --- Step 6: Adjust Parameterization with Reference Angle (if provided) ---
        if ref_angle is not None:
            if verbose:
                print("Adjusting parameterization using reference angle...")
            # As an example, map u_fine from [0, 1] to [0, 2π], shift by ref_angle[3], and map back.
            u_scaled = u_fine * 2 * np.pi
            u_shifted = (u_scaled + ref_angle[3]) % (2 * np.pi)
            u_fine = u_shifted / (2 * np.pi)
            # Re-evaluate the spline with the adjusted parameters.
            curve_eval = splev(u_fine, tck)
            curve_points = np.stack(curve_eval, axis=-1)
        
        if verbose:
            print("B-spline fitting complete.")
            print("Spline knot vector (from tck):")
            print(tck[0])
        
        # Return the evaluated curve, the tck (which includes knots, control points, and degree),
        # the fine parameter values, and the refined initial knots.
        return curve_points, tck, u_fine, final_knots

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
        
        # Possibly remove an outlier knot if it is too far
        still_outlier_knots = True
        knots_removed = 0
        while still_outlier_knots:
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

            if max_dist > 1.5 * np.median(knot_dists) and knots_removed < 5:
                knots_removed += 1
                final_knots = np.delete(final_knots, idx2, axis=0)
            else:
                still_outlier_knots = False
            

        # Build final spline
        loop_final_knots = fhf.loop_knots(final_knots)
        tt, curve = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')
        _, curve_pre = fhf.get_curve_from_knots(loop_final_knots, 'eq_vel')

        # Shift 'tt' to align with reference angles
        if ref_angle is not None and tt is not None:
            print("entered")
            tt = tt * 2 * np.pi
            tt_shifted = (tt + ref_angle[3]) % (2 * np.pi)

            # Check the first slope sign
            tt_diff = np.diff(tt_shifted)
            angle_diff = np.diff(ref_angle)
            # if np.sign(tt_diff[0]) != np.sign(angle_diff[0]):
            #     tt_shifted = np.flip(tt_shifted)
            #     curve = np.flip(curve, axis=0)
            tt = tt_shifted


        return curve, curve_pre, tt

    @staticmethod
    def plot_initial_knots(data_points, init_knots, session_idx, session, save_path=None):
        """
        Plots initial knots in 3D for debugging/visualization.
        """
     

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

        def interpolate_principal_curve(principal_curve, points_per_unit_distance=1000):
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
        if pc_interp is not None:
            tree = KDTree(pc_interp)
            distances, _ = tree.query(embeddings, k=1)
            mean_distance = np.mean(distances)
            return mean_distance, distances

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
        

        # Convert indices to a list (handles slices or list of indices)
        if isinstance(indices, slice):
            start = indices.start or 0
            stop = indices.stop or len(decoded_var)
            step = indices.step or 1
            indices = range(start, stop, step)
        elif not hasattr(indices, '__iter__'):
            raise TypeError("indices must be a slice or an iterable of integers")
        else:
            indices = list(indices)

        # Ensure same length
        if len(decoded_var) != len(behav_var):
            raise ValueError("decoded_var and behav_var must have the same length")

        # Extract data for the specified indices
        decoded_subset = [decoded_var[i] for i in indices]
        behav_subset = [behav_var[i] for i in indices]
        x_values = list(indices)

        # Plot
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_facecolor('white')

        # Plot decoded_var
        if legend_labels is not None and len(legend_labels) == 2:
            plt.plot(x_values, decoded_subset, label=legend_labels[0], color='red', linestyle='-')
            plt.plot(x_values, behav_subset, label=legend_labels[1], color='black', linestyle='-')
            plt.legend(loc="best")
        else:
            plt.plot(x_values, decoded_subset, color='red', linestyle='-')
            plt.plot(x_values, behav_subset, color='black', linestyle='-')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel1)
        plot_title = title
        if session_idx is not None:
            plot_title += f" - Session {session_idx}"
        plt.title(plot_title)
        ax.grid(False)
        plt.tight_layout()

        # Save the plot if save_path is specified
        if save_path:
            filename = f"decoded_var_{behav_var_name}.png" if behav_var_name else "decoded_var_plot.png"
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            if pdf:
                pdf.savefig()
            print(f"[INFO] Decoded vs. True plot saved to {full_path}")

        plt.show()
        plt.close()

    @staticmethod
    def plot_Hs_over_laps(
        est_H=None,
        decode_H=None,
        lap_number=None,
        session_idx=None,
        session=None,
        save_path=None,
        tag=None,
        SI_score=None,
        decode_err=None,
        pdf=None,
    ):

        if est_H is None or decode_H is None or lap_number is None:
            raise ValueError("est_H, decode_H, and lap_number must all be provided.")

        est_H = np.asarray(est_H)
        decode_H = np.asarray(decode_H)
        lap_number = np.asarray(lap_number)

        # Validate shapes
        if est_H.ndim != 1 or decode_H.ndim != 1 or lap_number.ndim != 1:
            raise ValueError("est_H, decode_H, and lap_number must be 1D arrays.")

        # Truncate to matching length
        min_length = min(len(est_H), len(decode_H), len(lap_number))
        est_H_trimmed = est_H[:min_length]
        decode_H_trimmed = decode_H[:min_length]
        lap_trimmed = lap_number[:min_length]

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(lap_trimmed, est_H_trimmed, label='est_H', color='blue', marker='o', linestyle='-')
        plt.plot(lap_trimmed, decode_H_trimmed, label='decode_H', color='red', marker='x', linestyle='--')

        ax = plt.gca()

        # Compute averages for annotations
        avg_est_H = np.mean(est_H_trimmed)
        avg_decode_H = np.mean(decode_H_trimmed)

        annotation_text = f'Avg est_H: {avg_est_H:.2f}\nAvg decode_H: {avg_decode_H:.2f}'
        if SI_score is not None:
            annotation_text += f'\nSI Score: {SI_score:.2f}'
        if decode_err is not None:
            annotation_text += f'\nDecode Error: {decode_err:.2f}'

        # Add text annotation
        ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

        # Axes labels and title
        plt.xlabel('Lap Number', fontsize=14)
        plt.ylabel('H Value', fontsize=14)
        title_str = 'est_H and decode_H Values Over Laps'
        if session_idx is not None and session is not None:
            # e.g. "Session 12: Rat 101, Day 3, Epoch m1"
            title_str += f'\nSession {session_idx}: Rat {session.rat}, Day {session.day}, Epoch {session.epoch}'
        if tag:
            title_str += f', Tag {tag}'
        plt.title(title_str, fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Save or show
        if save_path:
            # Construct subdirectory
            dir_components = [save_path, 'h_plots']
            if session_idx is not None:
                dir_components.append(f'session_{session_idx}')
            dir_path = os.path.join(*dir_components)
            os.makedirs(dir_path, exist_ok=True)

            # Construct filename
            filename = f"h_over_laps_{tag}.png" if tag else "h_over_laps.png"
            full_path = os.path.join(dir_path, filename)
            
            plt.savefig(full_path, dpi=300)
            if pdf:
                pdf.savefig()
            plt.close()
            print(f"[INFO] Saved Hs-over-laps plot to {full_path}")
        else:
            plt.show()

    @staticmethod
    def plot_Hs_over_laps_interactive(
        est_H=None,
        decode_H=None,
        lap_number=None,
        behav_var=None,
        session_idx=None,
        session=None,
        save_path=None,
        tag=None,
        SI_score=None,
        decode_err=None,
        mean_diff=None,
        std_diff=None,
        behav_var_name=None,
        pdf=None,
    ):
        """
        Plots two H values against lap numbers using Plotly for interactivity, with markers colored based on a behavioral variable.

        Parameters:
        - est_H (np.ndarray): Array of the first H values to plot.
        - decode_H (np.ndarray): Array of the second H values to plot.
        - lap_number (np.ndarray): Array of lap numbers corresponding to each H value.
        - behav_var (np.ndarray): Array of behavioral variable values to map to colors.
        - session_idx (int, optional): Session index for labeling purposes.
        - session (object, optional): Session object containing metadata.
        - save_path (str, optional): Directory path to save the plot HTML file.
        - tag (str, optional): Tag to include in the saved plot's filename.
        - SI_score (float, optional): Score to annotate on the plot.
        - decode_err (float, optional): Decoding error to annotate on the plot.
        - mean_diff (float, optional): Mean difference to annotate on the plot.
        - std_diff (float, optional): Standard deviation difference to annotate on the plot.
        Returns:
        - None
        """
        
        # Input validation
        if est_H is None or decode_H is None or lap_number is None:
            raise ValueError("est_H, decode_H, and lap_number must all be provided.")
        
        est_H = np.asarray(est_H)
        decode_H = np.asarray(decode_H)
        lap_number = np.asarray(lap_number)
        behav_var = np.asarray(behav_var)
        
        if est_H.ndim != 1 or decode_H.ndim != 1 or lap_number.ndim != 1:
            raise ValueError("est_H, decode_H, lap_number, and behav_var must all be 1-dimensional arrays.")
        
        # Ensure behav_var matches the data length
        # if behav_var is not None:
        #     min_length = min(len(est_H), len(decode_H), len(lap_number))
        #     behav_var_trimmed = behav_var[:min_length]
        # else:
        min_length = min(len(est_H), len(decode_H), len(lap_number))

        est_H_trimmed = est_H[:min_length]
        decode_H_trimmed = decode_H[:min_length]
        lap_trimmed = lap_number[:min_length]

        
        # Compute average values for annotations
        avg_est_H = np.mean(est_H_trimmed)
        avg_decode_H = np.mean(decode_H_trimmed)
        
        # Define color scale
        colorscale = 'Viridis'  # You can choose other Plotly colorscales like 'Cividis', 'Plasma', etc.
        
        # Define color range based on behav_var
        if behav_var is not None:
            cmin = np.min(behav_var)
            cmax = np.max(behav_var)
        
        # Create interactive plot with color mapping
        trace_est_H = go.Scatter(
            x=lap_trimmed,
            y=est_H_trimmed,
            mode='lines+markers',
            name='est_H Value',
            line=dict(color='blue'),
            marker=dict(symbol='circle', size=8)
        )
        if behav_var is not None:
            trace_decode_H = go.Scatter(
                x=lap_trimmed,
                y=decode_H_trimmed,
                mode='markers+lines',
                name='decode_H Value',
                line=dict(color='red', dash='dash'),
                marker=dict(
                    symbol='x',
                    size=8,
                    color=behav_var,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    showscale=False  
                )
            )
        else:
            trace_decode_H = go.Scatter(
                x=lap_trimmed,
                y=decode_H_trimmed,
                mode='markers+lines',
                name='decode_H Value',
                line=dict(color='red', dash='dash'),
                marker=dict(
                    symbol='x', size=8)
            )
        
        data = [trace_est_H, trace_decode_H]
        
        # Prepare annotation text
        annotation_text = f'Avg est_H: {avg_est_H:.2f}<br>Avg decode_H: {avg_decode_H:.2f}'
        if SI_score is not None:
            annotation_text += f'<br>SI Score: {SI_score:.2f}'
        if decode_err is not None:
            annotation_text += f'<br>Decode Error: {decode_err:.2f}'
        if mean_diff is not None:
            annotation_text += f'<br>Mean Diff: {mean_diff:.2f}'
        if std_diff is not None:
            annotation_text += f'<br>Std Diff: {std_diff:.2f}'
        
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

    @staticmethod
    def plot_Hs_moving_avg(
        est_H=None,
        decode_H=None,
        behav_var=None,
        behav_var_name=None,
        session_idx=None,
        session=None,
        save_path=None,
        tag=None,
        window_size=5,
        SI_score=None,
        decode_err=None,
        pdf=None,
    ):
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
            pdf.savefig()
            plt.close(fig)  # Close the figure to free memory
            print(f"Saved plot to {full_save_path}")
        else:
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
    Encapsulates the main analysis steps,
    using the utility methods from CEBRAUtils.
    """

    def __init__(self, session_choose=True, max_num_reruns=1, run_persistent_homology=False,include_land_off=False,whole_trial_embeddings=False,save_folder=None,trial_type='default',data_source="",temperature_list=None):
        """
        Loads data, sets up configuration, etc.
        """
        import scipy.io

        # --- Config parameters ---
        self.session_choose = session_choose
        if self.session_choose:
            self.landmark_sessions = []
            self.optic_flow_sessions = [35]
        else:
            self.landmark_num_trials = 0
            self.landmark_control_point = 25
            self.optic_flow_num_trials = 1
            self.optic_flow_control_point = 37

        self.run_persistent_homology = run_persistent_homology
        self.max_num_reruns = max_num_reruns
        self.include_land_off = include_land_off
        self.whole_trial_embeddings = whole_trial_embeddings
        self.save_folder = save_folder
        self.trial_type = trial_type
        self.data_source = data_source
        self.temperature_list = temperature_list

        self.sessions_data = []

        # path_optic_flow = os.path.join(
        #     '/Users/devenshidfar/Desktop/Masters/NRSC_510B/',
        #     'cebra_control_recal/mat_code_and_data/',
        #     'data/NN_opticflow_dataset.mat'
        # )
        path_optic_flow = os.path.join(
            '/Users/devenshidfar/Desktop/unclustered_processed/unclustered_all_913.mat'
        )
        

        path_landmark = os.path.join(
            '/Users/devenshidfar/Desktop/Masters/NRSC_510B/',
            'cebra_control_recal/mat_code_and_data/',
            'data/expt_landmark.mat'
        )

        path_unclustered = os.path.join(
            '/Users/devenshidfar/Desktop/unclustered.mat'
        )

        if callable(data_source):
            # If the user passed in a function
            self.expt = data_source()
        self.expt_optic_flow = CEBRAUtils.load_optic_flow_data(path_optic_flow)
        self.expt_landmark = CEBRAUtils.load_landmark_data(path_landmark)
        # if data_source == "unclustered":
        #     self.expt_unclustered = CEBRAUtils.load_unclustered_data(path_unclustered=path_unclustered)

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
        self.rm_low_vel = False
        self.bin_sizes = [1]
        self.max_num_reruns = max_num_reruns

        # For final data collection
        self.all_neural_data_full_trial = []
        self.all_neural_data_land_on = []
        self.all_neural_data_land_off = []
        self.all_embeddings_3d = []
        self.all_high_vel_mask = []
        self.H0_value = []
        self.H1_value = []
        self.all_betti_0 = []
        self.all_betti_1 = []
        self.all_principal_curves_3d = []
        self.all_curve_params_3d = []
        self.all_binned_hipp_angle = []
        self.all_binned_true_angle = []
        self.all_binned_est_gain = []
        self.all_vel = []
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
        # self.all_mse_decode_vs_true = []
        self.all_mean_H_difference = []
        self.all_std_H_difference = []
        self.expt_file = []

    def append_nan_values(self, session_idx, session):
        """
        Appends a SessionData instance with NaN values for missing data.
        """
        session_data = SessionData(
            expt_file=session,
            specifier=np.nan,
            neural_data_fit=np.nan,
            neural_data_low_vel=np.nan,
            embeddings_low_vel=np.nan,

            neural_data_full_trial=np.nan,
            neural_data_land_on=np.nan,
            neural_data_land_off=np.nan,
            embeddings_3d=np.nan,
            high_vel_mask=np.nan,
            H0_value=np.nan,
            H1_value=np.nan,
            betti_0=np.nan,
            betti_1=np.nan,
            principal_curves_3d=np.nan,
            curve_params_3d=np.nan,
            binned_hipp_angle=np.nan,
            binned_true_angle=np.nan,
            binned_est_gain=np.nan,
            binned_vel=np.nan,
            decoded_angles=np.nan,
            lap_decode_H = np.nan,
            lap_est_gain = np.nan,
            lap_number = np.nan,
            hipp_lap_number = np.nan,
            lap_vel = np.nan,
            filtered_decoded_angles_unwrap=np.nan,
            decode_H=np.nan,
            session_idx=session_idx,
            rat=session.rat,
            day=session.day,
            epoch=session.epoch,
            num_skipped_clusters=np.nan,
            num_used_clusters=np.nan,
            avg_skipped_cluster_iq=np.nan,
            avg_used_cluster_iq=np.nan,
            mean_dist_to_spline=np.nan,
            dist_to_spline=np.nan,
            mean_angle_diff=np.nan,
            shuffled_mean_angle_diff=np.nan,
            SI_score_hipp=np.nan,
            SI_score_true=np.nan,
            mean_H_difference=np.nan,
            std_H_difference=np.nan
        )
        self.sessions_data.append(session_data)

    
    def get_results_dict(self):
        """
        Returns the final results of the analysis as a structured dictionary
        where each session is a separate struct.
        """
        # Convert each SessionData instance to a dictionary
        sessions_dict_list = [asdict(session) for session in self.sessions_data]
        

        data_dict = {
            'sessions': sessions_dict_list,
            'specifier': "full_trial_included" if self.include_land_off else "only_vis_cue_included",
        }
        
        return data_dict


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

        i__unclustered_trials = 0
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
                        control_count += 1
                        # Skip control sessions
                        if control_count <= control_point:
                            print(f"Skipping session {session_idx + 1} (control count <= control_point).")
                            continue
                        elif control_count > (control_point + num_trials):
                            print("[INFO] Reached desired number of trials. Exiting session loop.")
                            break

                        print(f"\n[INFO] Processing session {session_idx}/{len(expt)} in {expt_name} experiments")
                        print(f"Rat: {session.rat}, Day: {session.day}, Epoch: {session.epoch}")

                        session_base_path = os.path.join(
                        '/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results',
                        self.save_folder,
                        expt_name,
                        f'rat_{session.rat}',
                        f'session_{session_idx}'
                        )   
                        os.makedirs(session_base_path, exist_ok=True)
                    
                        SI_plots_path = os.path.join(session_base_path, 'SI_Plots',self.trial_type)
                        anim_save_path = os.path.join(session_base_path, '3d_Animations',self.trial_type)
                        spectrogram_path = os.path.join(session_base_path, 'Spatial_Spectrograms',self.trial_type)
                        param_plot_path = os.path.join(session_base_path, 'Param_Plots',self.trial_type)
                        H_plot_path = os.path.join(session_base_path, 'H_Plots',self.trial_type)
                        pers_hom_path = os.path.join(session_base_path, 'Pers_Hom_Plots',self.trial_type)


                        paths_to_create = [SI_plots_path, anim_save_path, spectrogram_path, param_plot_path, H_plot_path, pers_hom_path]
                        for path in paths_to_create:
                            os.makedirs(path, exist_ok=True)

                        ros_data = session.rosdata
                        start_time = ros_data.startTs
                        end_time = ros_data.stopTs
                        land_off_time_sec = int(np.floor((ros_data.landOffTime - start_time) / 1e6)) # Floor it so we have it as integer
                        enc_times = np.array(ros_data.encTimes - start_time)
                        # Keep only those time points before landmarks are turned off AND divide by microseconds to get in seconds
                        enc_times = enc_times / 1e6
                        print(np.max(enc_times))
                        vel = np.array(ros_data.vel)
                        # Cutoff the velocity points accordingly as well
                        # Only finite values for times and velocity
                        valid_idx = np.isfinite(enc_times) & np.isfinite(vel)
                        enc_times = enc_times[valid_idx]
                        
                        vel = vel[valid_idx]
                        est_gain_filtered = np.array(ros_data.estGain)[valid_idx]
                        hipp_angle_filtered = np.array(ros_data.hippAngle)[valid_idx]
                        true_angle_filtered = np.array(ros_data.encAngle)[valid_idx]
                        rel_angle_filtered = np.array(ros_data.relAngle)[valid_idx]

                        bins = np.arange(enc_times[0], enc_times[-1] + bin_size, bin_size)
                        if len(bins) < 2:
                            print("[WARNING] Not enough bins after filtering for high velocity. Skipping session.")
                            continue
                        try:
                            binned_est_gain, _, _ = stats.binned_statistic(
                                enc_times, est_gain_filtered, statistic='mean', bins=bins
                            )
                            binned_hipp_angle, _, _ = stats.binned_statistic(
                                enc_times, hipp_angle_filtered, statistic='mean', bins=bins
                            )
                            binned_true_angle, _, _ = stats.binned_statistic(
                                enc_times, true_angle_filtered, statistic='mean', bins=bins
                            )
                            binned_vel, _, _ = stats.binned_statistic(
                                enc_times, vel, statistic='mean', bins=bins
                            )
                            binned_rel_angle, _, _ = stats.binned_statistic(
                                enc_times, rel_angle_filtered, statistic='mean', bins=bins
                            )
                        
                            # Define the high velocity mask
                            high_vel_mask = binned_vel > self.vel_threshold

                            # Set behavioral variable values to NaN where the high velocity mask is False
                            binned_est_gain[~high_vel_mask] = np.nan
                            binned_hipp_angle[~high_vel_mask] = np.nan
                            binned_true_angle[~high_vel_mask] = np.nan
                            binned_rel_angle[~high_vel_mask] = np.nan
            
                        except ValueError as e:
                            # Catch "Bin edges must be unique" 
                            print(f"[WARNING] Binning failed for session {session_idx}. Exception: {e}")
                            # Skip this session and continue with the next
                            continue

                        #Identify invalid bins the same way
                        # valid_bins = (
                        #     ~np.isnan(binned_hipp_angle) &
                        #     ~np.isnan(binned_true_angle) &
                        #     ~np.isnan(binned_est_gain) &
                        #     ~np.isnan(binned_vel)
                        # )

                        # # Instead of removing them, I set invalid bins to NaN:
                        # if not np.all(valid_bins):
                        #     binned_hipp_angle[~valid_bins] = np.nan
                        #     binned_true_angle[~valid_bins] = np.nan
                        #     binned_est_gain[~valid_bins] = np.nan
                        #     binned_vel[~valid_bins] = np.nan
                        #     binned_rel_angle[~valid_bins] = np.nan
                        

                        # Now interpolate the NaNs so the arrays still have the same length:
                        binned_hipp_angle = CEBRAUtils.linear_interpolate_nans_1d(binned_hipp_angle)
                        binned_true_angle = CEBRAUtils.linear_interpolate_nans_1d(binned_true_angle)
                        binned_est_gain   = CEBRAUtils.linear_interpolate_nans_1d(binned_est_gain)
                        binned_vel   = CEBRAUtils.linear_interpolate_nans_1d(binned_vel)
                        binned_rel_angle  = CEBRAUtils.linear_interpolate_nans_1d(binned_rel_angle)

                        # Filter spike times
                        all_spikes_full = []
                        all_spikes_train = []
                        skipped_clusters = 0
                        used_clusters = 0
                        used_cluster_iq_list = []
                        skipped_cluster_iq_list = []
                        if(self.data_source == "clustered"):
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
                                # include_spikes = vel_at_spikes > self.vel_threshold
                                # spike_times_sec_high_vel = spike_times_sec[include_spikes]
                                spike_times_sec_high_vel = spike_times_sec
                                if len(spike_times_sec_high_vel) == 0:
                                    continue
                                
                                try:
                                    binned_spikes_full, _, _ = stats.binned_statistic(
                                        spike_times_sec,
                                        np.ones_like(spike_times_sec),
                                        statistic='sum',
                                        bins=bins
                                    )
                                    all_spikes_full.append(binned_spikes_full)

                                    binned_spikes_train, _, _ = stats.binned_statistic(
                                        spike_times_sec_high_vel,
                                        np.ones_like(spike_times_sec_high_vel),
                                        statistic='sum',
                                        bins=bins
                                    )
                                    all_spikes_train.append(binned_spikes_train)

                                except ValueError as e:
                                    # Catch "Bin edges must be unique"
                                    print(f"[WARNING] Binning failed for session {session_idx}. Exception: {e}")
                                    # Skip this session and continue with the next
                                    continue

                        elif (self.data_source == "unclustered"):
                            use_all_tetrodes = False # Flag to decide whether or not to use all tetrodes or just ones in final processing
                            all_ttnums = set()
                            for cluster in session.clust: # Go through clusters in used clusters
                                all_ttnums.add(cluster.ttnum) # And add the ones that made it to final processing
                            for tetrode_data in session.unclustered: # Now go through tetrode data
                                if use_all_tetrodes or tetrode_data.ttnum in all_ttnums: # Check if that tetrode is in final processing
                                    spike_times_sec = (tetrode_data.ts - start_time) / 1e6

                                    plt.hist(tetrode_data.ts/1e6, bins=30, edgecolor='black')
                                    vel_at_spikes = tetrode_data.vel

                                    spike_times_sec_high_vel = spike_times_sec
                                    # Binning code
                                    binned_spikes_full, _, _ = stats.binned_statistic(
                                        spike_times_sec,
                                        np.ones_like(spike_times_sec),
                                        statistic='sum',
                                        bins=bins
                                    )
                                    all_spikes_full.append(binned_spikes_full)

                                    binned_spikes_train, _, _ = stats.binned_statistic(
                                        spike_times_sec_high_vel,
                                        np.ones_like(spike_times_sec_high_vel),
                                        statistic='sum',
                                        bins=bins
                                    )
                                    all_spikes_train.append(binned_spikes_train)
                        
                        else:
                            raise Exception("Incorrect data source")

        
                        # Stats for cluster
                        used_cluster_iq = np.asarray(used_cluster_iq_list)
                        skipped_cluster_iq = np.asarray(skipped_cluster_iq_list)
                        num_skipped_cluster = len(skipped_cluster_iq)
                        num_used_cluster = len(used_cluster_iq)
                        avg_skipped_cluster_iq = np.mean(skipped_cluster_iq) if len(skipped_cluster_iq) else np.nan
                        avg_used_cluster_iq = np.mean(used_cluster_iq) if len(used_cluster_iq) else np.nan

                        if not all_spikes_full:
                            print("[WARNING] No valid spike data after filtering. Skipping session.")
                            continue

                        # Build neural data
                        neural_data_full_trial = np.array(all_spikes_full).T
                        neural_data_train_trial = np.array(all_spikes_train).T
                        neural_data_land_off = neural_data_full_trial[land_off_time_sec:,:]
                        neural_data_land_on = neural_data_full_trial[:land_off_time_sec,:]

                        print("neural_data_full_trial shape:", neural_data_full_trial.shape)  # Full dataset
                        print("neural_data_land_off shape:", neural_data_land_off.shape)  # After slicing
                        print("neural_data_land_on shape:", neural_data_land_on.shape)  # After slicing
                        
                        num_bins_neural = neural_data_full_trial.shape[0]
                        num_bins_behavior = len(binned_est_gain)
                        if num_bins_neural != num_bins_behavior:
                            min_bins = min(num_bins_neural, num_bins_behavior)
                            neural_data_full_trial= neural_data_full_trial[:min_bins, :]
                            binned_est_gain = binned_est_gain[:min_bins]
                            binned_hipp_angle = binned_hipp_angle[:min_bins]
                            binned_true_angle = binned_true_angle[:min_bins]
                            binned_vel = binned_vel[:min_bins]
                            bins = bins[:min_bins]

                        
                        neural_data_fit = neural_data_train_trial[high_vel_mask, :]
                        neural_data_low_vel = neural_data_full_trial[~high_vel_mask,:]
                    

                        print(f"Shape of low vel neural data array: {neural_data_low_vel.shape}")

                        # For reruns if SI < threshold
                        pdf_filename = os.path.join(session_base_path, f'session_{session_idx}.pdf')
                        os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
                        pdf = PdfPages(pdf_filename)

                        best_embeddings_3d = None
                        best_SI_score_hipp = -999

                        # Attempt multiple runs if SI < some threshold
                        skip_session = False
                        for temp in self.temperature_list:
                            output_dimension = 3
                            rerun_count = 0
                            while rerun_count < self.max_num_reruns:
                                if self.include_land_off and self.whole_trial_embeddings: # If you want to include the data points after the landmarks/optic flow turned off or not
                                    embeddings_high_dim, embeddings_low_vel = CEBRAUtils.apply_cebra(
                                        neural_data_fit=neural_data_fit,
                                        neural_data_embeddings=neural_data_fit,
                                        neural_data_low_vel=neural_data_low_vel,
                                        output_dimension=output_dimension,
                                        temperature=temp
                                    )
                                elif not self.include_land_off and self.whole_trial_embeddings:
                                    embeddings_high_dim, embeddings_low_vel = CEBRAUtils.apply_cebra(
                                        neural_data_fit=neural_data_land_on,
                                        neural_data_embeddings=neural_data_full_trial,
                                        neural_data_low_vel=neural_data_low_vel,
                                        output_dimension=output_dimension,
                                        temperature=temp
                                    )
                                elif not self.include_land_off and not self.whole_trial_embeddings:
                                    embeddings_high_dim, embeddings_low_vel = CEBRAUtils.apply_cebra(
                                            neural_data_fit=neural_data_land_on,
                                            neural_data_embeddings=neural_data_land_on,
                                            neural_data_low_vel=neural_data_low_vel,
                                            output_dimension=output_dimension,
                                            temperature=temp
                                    )

                                embeddings_3d = embeddings_high_dim.copy()


                                full_embeddings = np.empty((neural_data_full_trial.shape[0],embeddings_3d.shape[1]))
                                full_embeddings[:] = np.nan
                                full_embeddings[high_vel_mask,:] = embeddings_3d
                                embeddings_3d_high_vel = full_embeddings

                                embeddings_3d = CEBRAUtils.linear_interpolate_nans_2d(embeddings_3d_high_vel)


                                print("shape of embeddings after fitting")
                                print(embeddings_3d.shape)
                                print("current hip angle shape")
                                print(binned_hipp_angle.shape)
                                print("embeddings low vel shape")
                                print(embeddings_low_vel.shape)


                                # Concatenate according to the size of embeddings_3d decided by the type of trial (land_on,land_off, etc.)
                                binned_hipp_angle = binned_hipp_angle[:embeddings_3d.shape[0]]
                                binned_true_angle = binned_true_angle[:embeddings_3d.shape[0]]
                                binned_est_gain = binned_est_gain[:embeddings_3d.shape[0]]
                                binned_vel = binned_vel[:embeddings_3d.shape[0]]

            
                                # Outlier removal
                                if self.rm_outliers:
                                    # Get full-length boolean mask
                                    inlier_mask_3d = CEBRAUtils.nt_TDA_mask(embeddings_3d,verbose=False)
                                    outlier_mask_3d = ~inlier_mask_3d

                                    # Set outliers to NaN 
                                    embeddings_3d[outlier_mask_3d, :] = np.nan

                                    # The binned_x arrays correspond 1-to-1 with embeddings_3d
                                    binned_hipp_angle[outlier_mask_3d] = np.nan
                                    binned_true_angle[outlier_mask_3d] = np.nan
                                    binned_est_gain[outlier_mask_3d] = np.nan
                                    binned_vel[outlier_mask_3d] = np.nan

                                    # Interpolate so there are no NaNs but the same shape
                                    embeddings_3d = CEBRAUtils.linear_interpolate_nans_2d(embeddings_3d)
                                    binned_hipp_angle = CEBRAUtils.linear_interpolate_nans_1d(binned_hipp_angle)
                                    binned_true_angle = CEBRAUtils.linear_interpolate_nans_1d(binned_true_angle)
                                    binned_est_gain = CEBRAUtils.linear_interpolate_nans_1d(binned_est_gain)
                                    binned_vel = CEBRAUtils.linear_interpolate_nans_1d(binned_vel)

                                if embeddings_3d.shape[0] < 201:
                                    print(f"length of embeddings is: {embeddings_3d.shape[0]}, skipping session {session_idx}")
                                    skip_session = True
                                    break

                                # Run persistent homology

                                if(self.run_persistent_homology):

                                    H0_value, H1_value, betti_0, betti_1 = CEBRAUtils.run_persistent_homology(
                                        embeddings=embeddings_3d,
                                        session_idx=session_idx,
                                        session=session,
                                        results_save_path=session_base_path,
                                        dimension=3
                                    )
                                else:
                                    H0_value = -1
                                    H1_value = -1
                                    betti_0 = -1
                                    betti_1 = -1


                                # Convert angles to radians
                                binned_true_angle_rad = np.deg2rad(binned_true_angle)
                                binned_hipp_angle_rad = np.deg2rad(binned_hipp_angle)

                                binned_true_angle_rad_unwrap = binned_true_angle_rad
                                binned_hipp_angle_rad_unwrap = binned_hipp_angle_rad

                                binned_true_angle_rad = (binned_true_angle_rad 
                                                        % (2 * np.pi))
                                binned_hipp_angle_rad = (binned_hipp_angle_rad 
                                                        % (2 * np.pi))
                        

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
                                elif SI_score_hipp < 0.8:
                                    print(f"[INFO] SI_score_hipp is {SI_score_hipp}. Retrying embedding.")
                                    rerun_count += 1

                            # If we tried max_num_reruns times, keep the best anyway

                            if skip_session:
                                print("[INFO] not enough embedding points "
                                        "Skipping the entire session and writing NaNs.")
                                print(session_idx)
                                print(session)
                                self.append_nan_values(session_idx,session)

                                # Continue to the next session if not enough embedding points
                                skip_session = False
                                continue

                            if best_embeddings_3d is not None:
                                embeddings_3d = best_embeddings_3d
                            else:
                                # fallback
                                embeddings_3d = embeddings_high_dim

                            

                            # Compute SI with true angle
                            SI_score_true = CEBRAUtils.compute_SI_and_plot(
                                embeddings=embeddings_3d,
                                behav_var=binned_true_angle_rad,
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

                            #Fit principal curve (SPUD)
                            fit_params = {
                                'dalpha': 0.005,
                                'knot_order': 'nearest',
                                'penalty_type': 'curvature',
                                'nKnots': 15,
                                'curvature_coeff': 100,
                                'len_coeff': 2,
                                'density_coeff': 2,
                                'delta': 0.1
                            }           

                            principal_curve_3d, principal_curve_3d_pre, curve_params_3d = CEBRAUtils.fit_spud_to_cebra(
                                embeddings=embeddings_3d,
                                ref_angle=binned_true_angle_rad_unwrap,
                                session_idx=session_idx,
                                session=session,
                                results_save_path=os.path.join(os.path.dirname(pdf_filename)),
                                fit_params=fit_params,
                                dimension_3d=1,
                                verbose=False
                            )

                            if principal_curve_3d is None:
                                print("[INFO] Detected knots were too close or"
                                      "Skipping the entire session and writing NaNs.")

                                self.append_nan_values(session_idx, session)

                                # Continue to the next session if principal curve is None 
                                continue

                            # Distance to principal curve
                            mean_dist_to_spline, distances_to_spline = CEBRAUtils.mean_dist_to_spline(embeddings_3d, principal_curve_3d)
                            print("dists to spline")
                            print(distances_to_spline[:20])

                            # Decode angles
                            decoded_angles, _ = CEBRAUtils.decode_hipp_angle_spline(
                                embeddings=embeddings_3d,
                                principal_curve=principal_curve_3d,
                                tt=curve_params_3d,
                                true_angles=binned_true_angle_rad_unwrap
                            )
                            # Shuffled
                            shuffled_true = np.random.permutation(binned_true_angle_rad)
                            shuffled_decoded_angles, _ = CEBRAUtils.decode_hipp_angle_spline(
                                embeddings=embeddings_3d,
                                principal_curve=principal_curve_3d,
                                tt=curve_params_3d,
                                true_angles=shuffled_true
                            )


                            decoded_angles_unwrap = np.unwrap(decoded_angles + binned_true_angle_rad_unwrap[3])
                            binned_hipp_angle_unwrap = np.unwrap(binned_hipp_angle)

                            print("range")
                            print(np.max(decoded_angles) - np.min(decoded_angles))
                            shuffled_decoded_angles_unwrap = np.unwrap(shuffled_decoded_angles + binned_true_angle_rad_unwrap[3])
                            decoded_angles = (decoded_angles_unwrap) % (2 * np.pi)
                            shuffled_decoded_angles = (shuffled_decoded_angles_unwrap) % (2 * np.pi)
                            
                            angle_diff = (decoded_angles_unwrap - binned_hipp_angle_rad_unwrap)
                            shuffled_angle_diff = (shuffled_decoded_angles_unwrap - binned_hipp_angle_rad_unwrap)
                            mean_angle_diff = np.mean(angle_diff)
                            shuffled_mean_angle_diff = np.mean(shuffled_angle_diff)

                            # Low-pass filter
                            filtered_decoded_angles_unwrap = CEBRAUtils.low_pass_filter(
                                angles=decoded_angles_unwrap, cutoff_frequency=0.2, filter_order=3, fs=1
                            )
                            filtered_decoded_angles_unwrap = savgol_filter(filtered_decoded_angles_unwrap, window_length=30, polyorder=2)

                            derivative_decoded_angle = CEBRAUtils.derivative_and_mv_avg(
                                data=filtered_decoded_angles_unwrap, window_size=60
                            )
                            derivative_true_angle = CEBRAUtils.derivative_and_mv_avg(
                                data=binned_true_angle_rad_unwrap, window_size=60
                            )
                            derivative_hipp_angle = CEBRAUtils.derivative_and_mv_avg(
                                data=binned_hipp_angle_rad_unwrap, window_size=60
                            )


                            # Save side-by-side plots in PDF
                            os.makedirs(param_plot_path, exist_ok=True)
                            CEBRAUtils.plot_decoded_var_and_true(
                                decoded_var=filtered_decoded_angles_unwrap,
                                behav_var=binned_hipp_angle_rad_unwrap,
                                save_path=param_plot_path,
                                session_idx=session_idx,
                                behav_var_name='Hipp',
                                pdf=pdf,
                                legend_labels=['Filtered Decoded', 'Hipp Angle']
                            )

            
                            CEBRAUtils.plot_embeddings_static(
                                embeddings = embeddings_3d,
                                embeddings_low_vel = embeddings_low_vel,
                                principal_curve = principal_curve_3d,
                                behav_var_name='Low Vel',  
                                behav_var=binned_hipp_angle_rad,
                                session_idx=session_idx,
                                save_path=anim_save_path
                            )


                            CEBRAUtils.plot_embeddings_interactive(
                                embeddings=embeddings_3d,
                                embeddings_low_vel=embeddings_low_vel,
                                principal_curve=principal_curve_3d,
                                behav_var_name="Low Velocity Embeddings",
                                behav_var=binned_hipp_angle_rad,
                                session_idx=session_idx,
                                html_filename="embeddings_3d.html",
                                save_path=anim_save_path
                            )




                            # 3D plot
                            # plot_file_path = os.path.join(anim_save_file, f"3d_plot_session_{session_idx}.png")


                            # Now compute decode_H
                            decode_H = derivative_decoded_angle / derivative_true_angle
                            decode_H_nan_count = np.isnan(decode_H).sum()
                            decode_H = np.clip(decode_H, -2, 2)

                            min_len = min(len(binned_est_gain), len(decode_H))
                            mean_H_diff = np.mean(np.abs(binned_est_gain[:min_len] - decode_H[:min_len]))
                            std_H_diff = np.std(binned_est_gain[:min_len] - decode_H[:min_len])

                            # Behav vars over lapss
                            lap_number, sorted_decode_H, sorted_lap_number = CEBRAUtils.get_var_over_lap(
                                var=decode_H, true_angle=binned_true_angle_rad_unwrap
                            )
                            _, sorted_H_est, _ = CEBRAUtils.get_var_over_lap(
                                var=binned_est_gain, true_angle=binned_true_angle_rad_unwrap
                            )
                            _, sorted_vel, _ = CEBRAUtils.get_var_over_lap(
                                var=vel, true_angle=binned_true_angle_rad_unwrap
                            )
                            # Hipp frame
                            hipp_lap_number, hipp_decode_H, hipp_sorted_lap_number = CEBRAUtils.get_var_over_lap(
                                var=decode_H, true_angle=binned_hipp_angle_rad_unwrap
                            )
                            # _, sorted_rel_angle, _ = CEBRAUtils.get_var_over_lap(
                            #     var=np.deg2rad(rel_angle_filtered[valid_bins][nan_mask_3d]), 
                            #     true_angle=binned_true_angle_rad_unwrap
                            # ) if len(rel_angle_filtered) else (None, None, None)

                            CEBRAUtils.plot_Hs_over_laps(
                                est_H=sorted_H_est,
                                decode_H=sorted_decode_H,
                                lap_number=sorted_lap_number,
                                session_idx=session_idx,
                                session=session,
                                save_path=H_plot_path,    
                                tag="no_ma",
                                SI_score=SI_score_hipp,
                                decode_err=mean_angle_diff,
                                pdf=pdf
                            )

                            CEBRAUtils.plot_Hs_over_laps_interactive(
                                est_H=sorted_H_est,
                                decode_H=sorted_decode_H,
                                lap_number=sorted_lap_number,
                                session_idx=session_idx,
                                behav_var=vel,
                                session=session,
                                save_path=H_plot_path,
                                tag='vel_no_ma',
                                SI_score=SI_score_hipp,
                                decode_err=mean_angle_diff,
                                mean_diff=mean_H_diff,
                                std_diff=std_H_diff,
                                behav_var_name="None"
                            )

                            decode_H_ma = CEBRAUtils.compute_moving_average(decode_H, window_size=20)

                            CEBRAUtils.plot_Hs_over_laps(
                                est_H=sorted_H_est,
                                decode_H=sorted_decode_H,
                                lap_number=sorted_lap_number,
                                session_idx=session_idx,
                                session=session,
                                save_path=H_plot_path,    
                                tag="ma_20_sec",
                                SI_score=SI_score_hipp,
                                decode_err=mean_angle_diff,
                                pdf=pdf
                            )

                            CEBRAUtils.plot_Hs_over_laps_interactive(
                                est_H=sorted_H_est,
                                decode_H=sorted_decode_H,
                                lap_number=sorted_lap_number,
                                session_idx=session_idx,
                                behav_var=vel,
                                session=session,
                                save_path=H_plot_path,
                                tag='vel_ma_20',
                                SI_score=SI_score_hipp,
                                decode_err=mean_angle_diff,
                                mean_diff=mean_H_diff,
                                std_diff=std_H_diff,
                                behav_var_name="None"
                            )

                            CEBRAUtils.plot_Hs_moving_avg(
                                est_H=sorted_H_est,
                                decode_H=sorted_decode_H,
                                behav_var=vel,
                                behav_var_name="vel",
                                session_idx=session_idx,
                                session=session,
                                save_path=H_plot_path,
                                tag='vel_ma_20',
                                window_size=5,
                                SI_score=SI_score_hipp,
                                decode_err=mean_angle_diff,
                                pdf=pdf
                            )



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


                            session_data = SessionData(
                                expt_file=session,
                                specifier=np.nan,
                                neural_data_full_trial=neural_data_full_trial,
                                neural_data_fit=neural_data_fit,
                                neural_data_low_vel=neural_data_low_vel,
                                embeddings_low_vel=embeddings_low_vel,
                                high_vel_mask = high_vel_mask,
                                neural_data_land_off=neural_data_land_off,
                                neural_data_land_on=neural_data_land_on,
                                embeddings_3d=embeddings_3d,
                                H0_value=H0_value,
                                H1_value=H1_value,
                                betti_0=betti_0,
                                betti_1=betti_1,
                                principal_curves_3d=principal_curve_3d,
                                curve_params_3d=curve_params_3d,
                                binned_hipp_angle=binned_hipp_angle_rad_unwrap,
                                binned_true_angle=binned_true_angle_rad_unwrap,
                                binned_est_gain=binned_est_gain, 
                                binned_vel=binned_vel,
                                decoded_angles=decoded_angles_unwrap,
                                filtered_decoded_angles_unwrap=filtered_decoded_angles_unwrap,
                                decode_H=decode_H,
                                lap_decode_H=sorted_decode_H,
                                lap_est_gain=sorted_H_est,
                                lap_vel=sorted_vel,
                                lap_number=lap_number,
                                hipp_lap_number=hipp_lap_number,
                                session_idx=session_idx,
                                rat=session.rat,
                                day=session.day,
                                epoch=session.epoch,
                                num_skipped_clusters=num_skipped_cluster,
                                num_used_clusters=num_used_cluster,
                                avg_skipped_cluster_iq=avg_skipped_cluster_iq,
                                avg_used_cluster_iq=avg_used_cluster_iq,
                                mean_dist_to_spline=mean_dist_to_spline,
                                dist_to_spline=distances_to_spline,
                                mean_angle_diff=mean_angle_diff,
                                shuffled_mean_angle_diff=shuffled_mean_angle_diff,
                                SI_score_hipp=best_SI_score_hipp,
                                SI_score_true=SI_score_true,
                                mean_H_difference=mean_H_diff,
                                std_H_difference=std_H_diff
                            )
                            
                            self.sessions_data.append(session_data)

                        pdf.close()

                    
                    except AttributeError as e:
                        print(f"[WARNING] Session {session_idx + 1} is invalid or missing. Skipping...")
                        print(f"Error: {e}")
                        continue

                    #end Expt loop

def main():
    """
    Entry point to run the entire analysis.
    """

    save_folder = 'thesis_results'

    #Run analysis with no including when landmarks/optic flow are off
    # analysis_train_land_on = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=False,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='train_land_on'
    # )

    #Run analysis with including when landmarks/optic flow off
    # analysis_full_trial = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=True,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='full_trial'
    # )

    # analysis_land_on = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=False,
    #     whole_trial_embeddings=False,
    #     save_folder=save_folder,
    #     trial_type='land_on'
    # )

    # # analysis_land_on.run_analysis()
    # analysis_full_trial.run_analysis()
    # # analysis_train_land_on.run_analysis()


    
    # # Get their results as dictionaries:

    # dict_analysis_land_on = analysis_land_on.get_results_dict()
    # dict_analysis_full_trial = analysis_full_trial.get_results_dict()
    # dict_analysis_train_land_on = analysis_train_land_on.get_results_dict()
    # combined_results = {
    #     'land_on': dict_analysis_land_on,
    #     'full_trial': dict_analysis_full_trial,
    #     'train_land_on': dict_analysis_train_land_on
    # }


    # analysis_full_trial_1 = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=True,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='unclustered',
    #     data_source='unclustered',
    #     temperature_list=[1]
    # )

    # analysis_full_trial_1.run_analysis()
    # dict_analysis_full_trial_1 = analysis_full_trial_1.get_results_dict()


    analysis_full_trial_2 = CEBRAAnalysis(
        session_choose=False,
        max_num_reruns=1,
        run_persistent_homology=False,
        include_land_off=True,
        whole_trial_embeddings=True,
        save_folder=save_folder,
        trial_type='full_trial_2',
        data_source='clustered',
        temperature_list=[1]
    )

    analysis_full_trial_2.run_analysis()
    dict_analysis_full_trial_2 = analysis_full_trial_2.get_results_dict()



    # analysis_full_trial_1 = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=True,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='full_trial_1',
    #     data_source='unclustered',
    #     temperature_list=[1]
    # )

    # analysis_full_trial_1.run_analysis()
    # dict_analysis_full_trial_1 = analysis_full_trial_1.get_results_dict()




    # analysis_full_trial_3 = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=True,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='unclustered_3',
    #     data_source='unclustered',
    #     temperature_list=[1]
    # )

    # analysis_full_trial_3.run_analysis()
    # dict_analysis_full_trial_3 = analysis_full_trial_3.get_results_dict()
    
    # analysis_full_trial_2 = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=True,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='full_trial_2'
    # )
    # analysis_full_trial_2.run_analysis()
    # dict_analysis_full_trial_2 = analysis_full_trial_2.get_results_dict()
    # analysis_full_trial_3 = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=True,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='full_trial_3'
    # )
    # analysis_full_trial_3.run_analysis()
    # dict_analysis_full_trial_3 = analysis_full_trial_3.get_results_dict()
    # analysis_full_trial_4 = CEBRAAnalysis(
    #     session_choose=False,
    #     max_num_reruns=1,
    #     run_persistent_homology=False,
    #     include_land_off=True,
    #     whole_trial_embeddings=True,
    #     save_folder=save_folder,
    #     trial_type='full_trial_4'
    # )
    # analysis_full_trial_4.run_analysis()
    # dict_analysis_full_trial_4 = analysis_full_trial_4.get_results_dict()

    combined_results = {
        'full_trial_1': dict_analysis_full_trial_1,
        'full_trial_2': dict_analysis_full_trial_2,
        # 'full_trial_4': dict_analysis_full_trial_4
    }
    
    # Combine them into a single dictionary
    # so that each run is stored under a different field/struct

    base_path = f'/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/results/'
    mat_filename = os.path.join(base_path, save_folder, f'{save_folder}_all_sessions_data.mat')
    savemat(mat_filename, combined_results)
    print(f"[INFO] Saved final data to {mat_filename}")

    print(f"[INFO] Successfully saved combined results")

if __name__ == "__main__":
    main()
