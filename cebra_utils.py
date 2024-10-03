import sys
import os
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


def fit_spud_to_cebra(embeddings_3d, nKnots=15, knot_order='wt_per_len', penalty_type='mult_len', length_penalty=10):
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

def create_rotating_3d_plot(embeddings_3d, session, anim_save_path, save_anim, principal_curve):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])
        ax.plot(principal_curve[:, 0], principal_curve[:, 1], principal_curve[:, 2], color='red', linewidth=2)
        
        ax.set_title(f"3D: Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
        ax.set_xlabel('Embedding Dimension 1')
        ax.set_ylabel('Embedding Dimension 2')
        ax.set_zlabel('Embedding Dimension 3')

        def rotate(angle):
            ax.view_init(elev=10., azim=angle)
            return scatter,

        anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50, blit=True)
        
        if(anim):
            if save_anim:
                anim.save(anim_save_path, writer='pillow', fps=30)
            else:
                plt.show()

        return anim
