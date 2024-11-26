'''
March 22nd 2019
Functions to fit a 1D piecewise linear spline to a pointcloud and use
it to do decoding.
'''


import numpy as np
import numpy.linalg as la
from pandas import cut
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

import angle_fns as af
import spud_code.shared_scripts.fit_helper_fns_custom as fhf
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn_extra.cluster import KMedoids
from functools import partial

class PiecewiseLinearFit:
    '''Fits a piecewise linear curve to passed data. The curve runs through
    a series of knots.'''

    def __init__(self, data_to_fit, params):
        self.data_to_fit = data_to_fit
        self.nDims = data_to_fit.shape[1]
        self.nKnots = params['nKnots']
        self.saved_knots = []

        # This sets the resolution at which the curve is sampled
        # Note that we don't care about these coordinates too much, since
        # they won't be used for decoding: they're just a way to generate the
        # curve so that we can compute fit error
        self.tt = np.arange(0, 1 + params['dalpha'] / 2., params['dalpha'])
        self.t_bins, self.t_int_idx, self.t_rsc = self.global_to_local_coords()

    def get_new_initial_knots(self, method='kmedoids'): #edited from Chaudhuri et al.
        '''Place the initial knots for the optimization to use.'''
        print(f"method: {method}")
        if method == 'dbscan':
            print("Doing DBSCAN initial clustering")
            clustering = DBSCAN(eps=0.1, min_samples=6).fit(self.data_to_fit)
            labels = clustering.labels_
            unique_labels = set(labels)
            unique_labels.discard(-1)  # Remove noise label
            cluster_centers = []
            print(f"DBSCAN labels: {labels}")
            print(f"Unique valid labels (excluding noise): {unique_labels}")

            for label in unique_labels:
                cluster_points = self.data_to_fit[labels == label]
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
            cluster_centers = np.array(cluster_centers)
            # Adjust the number of knots as needed
            if len(cluster_centers) >= self.nKnots:
                return cluster_centers[:self.nKnots]
            else:
                # Sample additional knots from high-density regions
                additional_knots = self.data_to_fit[np.random.choice(len(self.data_to_fit), self.nKnots - len(cluster_centers), replace=False)]
                return np.vstack([cluster_centers, additional_knots])
        elif method == 'kmeans':
            print("Doing K-Means initial clustering")
            kmeans = KMeans(n_clusters=self.nKnots, max_iter=3000).fit(self.data_to_fit)
            return kmeans.cluster_centers_
        elif method == 'kmedoids':
            print("Doing K-Medoids initial clustering")
            # Initialize K-Medoids with desired parameters
            kmedoids = KMedoids(n_clusters=self.nKnots, method='pam', metric='euclidean', random_state=42)
            kmedoids.fit(self.data_to_fit)
            print(f"cluster centers: {kmedoids.cluster_centers_}")
            return kmedoids.cluster_centers_
        else:
            print("hi")
            print('Unknown method')

    def order_knots(self, knots, method='nearest'):
        '''Order the initial knots so that we can draw a curve through them. 
        Start with a randomly chosen knot and then successively move to the
        "nearest" knot, where nearest can be determined by a specified method.'''

        ord_knots = np.zeros_like(knots)
        rem_knots = knots.copy()

        # Pick a random knot to start
        next_idx = np.random.choice(len(rem_knots))
        ord_knots[0] = rem_knots[next_idx]

        for i in range(1, len(knots)):
            rem_knots = np.delete(rem_knots, next_idx, axis=0)
            if method == 'nearest':
                # Nearest ambient distance
                next_idx = fhf.find_smallest_dist_idx(ord_knots[i - 1], rem_knots)
            elif method == 'wt_per_len':
                # Choose the closest as measured by density (wt_per_len)
                dists = np.linalg.norm(ord_knots[i - 1] - rem_knots, axis=1)
                r = np.min(dists)
                wts = np.array([np.sum(np.exp(-fhf.get_distances_near_line(ord_knots[i - 1],
                    k, self.data_to_fit, r) / r)) for k in rem_knots])
                # Used to be wts / (dists**alpha)
                wt_per_len = wts / (dists)
                next_idx = np.argmax(wt_per_len)
            ord_knots[i] = rem_knots[next_idx].copy()
        return ord_knots
    
    def compute_curvature(self, loop_knots):
        segments = loop_knots[1:] - loop_knots[:-1]  # Shape: (n_knots, n_dims)
        directions = segments / (np.linalg.norm(segments, axis=1)[:, np.newaxis]+1e-7) # Normalize
        delta_directions = directions[1:] - directions[:-1]  # Changes in direction
        curvature = np.sum(np.linalg.norm(delta_directions, axis=1)**2)  # Sum of squared changes
        return curvature
    
    def huber_loss(self, dists, delta):
            return np.where(
                dists <= delta,
                0.5 * dists ** 2,
                delta * (dists - 0.5 * delta)
    )

    def fit_data(self, fit_params, verbose=False):
        '''Main function to fit the data. Starting from the initial knots
        move them to minimize the distance of points to the curve, along with
        some (optional) penalty.'''
        print("fit_params in fit_data:", fit_params)

        save_dict = {'fit_params': fit_params}

        def cost_fn(flat_knots, fit_params,verbose=False):

            #print(f"fit params inside cost_fn: {fit_params}")
            # Reshape flat_knots to (nKnots, nDims)
            knots = np.reshape(flat_knots.copy(), (self.nKnots, self.nDims))
            
            # Generate looped knots
            loop_knots = fhf.loop_knots(knots)
            
            # Generate the fit curve based on local coordinates
            fit_curve = loop_knots[self.t_int_idx] + (loop_knots[self.t_int_idx + 1] - loop_knots[self.t_int_idx]) * self.t_rsc[:, np.newaxis]
            
            # Find nearest neighbors from data to the fit curve
            neighbgraph = NearestNeighbors(n_neighbors=1).fit(fit_curve)
            dists, inds = neighbgraph.kneighbors(self.data_to_fit)

            mean_dists = np.mean(dists)
            threshold = 1.5 * mean_dists
            filtered_dists = dists[dists < threshold]
            inds = inds[dists < threshold]

            
            # Compute data density using Kernel Density Estimation
            kde = KernelDensity(kernel='gaussian', bandwidth=2.0).fit(self.data_to_fit)
            log_density_data = kde.score_samples(self.data_to_fit)
            density_data = np.exp(log_density_data)
            weights = density_data / np.sum(density_data)
            
            # Compute weighted distances using Huber loss
            weighted_dists = filtered_dists.flatten() * weights[inds.flatten()]
            delta = fit_params.get('delta', 0.1)  # Adjust delta as needed
            
            # Huber loss computation
            huber_loss = np.where(
                weighted_dists <= delta,
                0.5 * weighted_dists ** 2,
                delta * (weighted_dists - 0.5 * delta)
            )
            


            # Base cost: sum of Huber losses
            cost = np.sum(huber_loss)
            if(verbose):
                print(f"dist penalty: {cost}")

            #print(f"PENALTY PARAMS: {fit_params['penalty_type']}")
            
            # Apply penalties based on penalty_type
            if fit_params['penalty_type'] == 'none':
                return cost
            elif fit_params['penalty_type'] == 'mult_len':
                cost *= self.tot_len(loop_knots)
                return cost
            elif fit_params['penalty_type'] == 'add_len':
                cost += fit_params['len_coeff'] * self.tot_len(loop_knots)
                return cost
            elif fit_params['penalty_type'] == 'curvature':
                curvature_penalty = fit_params['curvature_coeff'] * self.compute_curvature(loop_knots)
                if(verbose):
                    print(f"curvature penalty is: {curvature_penalty}")
                length_penalty = fit_params['len_coeff'] * self.tot_len(loop_knots)
                if(verbose):
                    print(f"length penalty is: {length_penalty}")
                # Density penalty
                log_density_knots = kde.score_samples(knots)
                if(verbose):
                    print(f"log density of knots: {log_density_knots}")
                density_penalty = fit_params['density_coeff'] * np.sum(-log_density_knots)
                if(verbose):
                    print(f"density penalty is: {density_penalty}")
                cost += curvature_penalty + length_penalty + density_penalty
                return cost
            else:
                raise ValueError(f"Unknown penalty type: {fit_params['penalty_type']}")

        init_knots = fit_params['init_knots']
        if(verbose):
            print("init_knots shape:", init_knots.shape)

        flat_init_knots = init_knots.flatten()
        if(verbose):
            print("flat_init_knots size:", flat_init_knots.size)

        bound_cost_fn = partial(cost_fn, fit_params=fit_params,verbose=verbose)

        fit_result = minimize(
            bound_cost_fn,
            flat_init_knots,
            method='Nelder-Mead',
            options={'maxiter': 100},
        )

        knots = np.reshape(fit_result.x.copy(), (self.nKnots, self.nDims))
        save_dict = {'knots': knots, 'err': fit_result.fun, 'init_knots': init_knots}
        self.saved_knots.append(save_dict)



    # Various utility functions
    def global_to_local_coords(self):
        '''tt is a global coordinate that runs from 0 to 1. But the curve is made
        up of a series of line segments which have local coordinates. So we want to break
        tt into equally spaced sets, each corresponding to one line segment. 
        Note that these coordinates aren't used for decoding, just to generate the curve,
        so that the rate at which they increase around the curve doesn't matter, as long 
        as we generate the curve at a decent resolution. '''

        # Equally spaced bins of tt, with t_bins[0] to t_bins[1] corresponding to
        # the first line segment and so on.
        t_bins = np.linspace(0, 1., self.nKnots + 1)
        # For each element in tt, figure out which bin it should lie in
        # Replace cut later if we want
        t_int_idx = cut(self.tt, bins=t_bins, labels=False, include_lowest=True)

        # Now get the local coordinates along each line segment
        # t_int_idx is the link between this and the global coordinate and will
        # tell us which linear function to apply.
        t_rsc = (self.tt - t_bins[t_int_idx]) / (t_bins[t_int_idx + 1] - t_bins[t_int_idx])

        return t_bins, t_int_idx, t_rsc

    def get_curve_from_knots_internal(self, inp_knots):
        '''Turn a list of knots into a curve, sampled at the pre-specified
        resolution.'''

        # Repeat the first knot at the end so we get a loop.
        loop_knots = fhf.loop_knots(inp_knots)
        return loop_knots[self.t_int_idx] + (loop_knots[self.t_int_idx + 1] -
            loop_knots[self.t_int_idx]) * self.t_rsc[:, np.newaxis]

    def distance_from_curve(self, inp_knots):
        '''Cost function to test a given set of knots.
        Assuming knots aren't looped around '''

        fit_curve = self.get_curve_from_knots_internal(inp_knots)
        neighbgraph = NearestNeighbors(n_neighbors=1).fit(fit_curve)
        dists, inds = neighbgraph.kneighbors(self.data_to_fit)
        cost = np.sum(dists)
        return cost

    def tot_len(self, loop_knot_list):
        ls_lens = la.norm(loop_knot_list[1:]-loop_knot_list[:-1],axis=1)
        return np.sum(ls_lens)


def fit_manifold(data_to_fit, fit_params):
    '''fit_params takes nKnots : number of knots, dalpha : resolution for
    sampled curve, knot_order : method to initially order knots, penalty_type : 
    penalty'''
    # fit_params is a superset of the initial params that PiecewiseLinearFit needs
    fitter =  PiecewiseLinearFit(data_to_fit, fit_params)
    unord_knots = fitter.get_new_initial_knots()
    init_knots = fitter.order_knots(unord_knots, method=fit_params['knot_order'])
    curr_fit_params = {'init_knots' : init_knots, 'penalty_type' : 
        fit_params['penalty_type'],  'len_coeff': fit_params['len_coeff'], 
        'curvature_coeff': fit_params['curvature_coeff']}
    fitter.fit_data(curr_fit_params)
    fit_results = dict(fit_params)
    fit_results['init_knots'] = init_knots
    # The fit class appends the results of each fit to a list called saved_knots
    # Here we're just using the class once, hence saved_knots[0]
    fit_results['final_knots'] = fitter.saved_knots[0]['knots']
    fit_results['fit_err'] = fitter.saved_knots[0]['err']
    fit_results['loop_final_knots'] = fhf.loop_knots(fitter.saved_knots[0]['knots'])
    fit_results['tt'], fit_results['curve'] = fhf.get_curve_from_knots(
        fit_results['loop_final_knots'], 'eq_vel')
    return fit_results

def decode_from_passed_fit(data_to_decode, fit_coords, fit_curve, ref_angles):
    #  When calling, trim fit_coords and fit_curve before passing so first and last entries 
    # aren't identical, though I suspect it won't matter (check this)
    # loop_tt, loop_curve = fit_results['tt'], fit_results['curve']

    unshft_coords = fhf.get_closest_manifold_coords(fit_curve, 
        fit_coords, data_to_decode)
    dec_angle, mse, shift, flip = af.shift_to_match_given_trace(unshft_coords,
        ref_angles)
    return dec_angle, mse
