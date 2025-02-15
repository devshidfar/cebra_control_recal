import sys
sys.path.append('/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/real_data/analysis_code')

from run_manifold_H import CEBRAUtils

import unittest
import numpy as np
from run_manifold_H import CEBRAUtils
from scipy import signal

class TestCEBRAUtils(unittest.TestCase):

    def test_linear_interpolate_nans_1d(self):
        # Create a 1D array with a NaN in the middle
        arr = np.array([1.0, np.nan, 3.0, 4.0])
        interpolated = CEBRAUtils.linear_interpolate_nans_1d(arr.copy())
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(interpolated, expected, rtol=1e-5)

    def test_linear_interpolate_nans_2d(self):
        # Create a 2D array with NaNs in some columns
        arr = np.array([[1.0, 2.0],
                        [np.nan, 4.0],
                        [3.0, np.nan],
                        [4.0, 8.0]])
        interpolated = CEBRAUtils.linear_interpolate_nans_2d(arr.copy())
        expected = np.array([[1.0, 2.0],
                             [2.0, 4.0],
                             [3.0, 6.0],
                             [4.0, 8.0]])
        np.testing.assert_allclose(interpolated, expected, rtol=1e-5)

    def test_derivative_and_mv_avg(self):
        # Use a simple array to check derivative and moving average
        data = np.array([1, 2, 4, 7, 11])
        result = CEBRAUtils.derivative_and_mv_avg(data, window_size=3)
        # Manually compute expected: first diff = [1,2,3,4], insert first element -> [1,1,2,3,4],
        # then moving average with kernel [1/3,1/3,1/3] in 'same' mode.
        diffs = np.diff(data)
        diffs = np.insert(diffs, 0, data[0])
        kernel = np.ones(3) / 3
        expected = np.convolve(diffs, kernel, mode='same')
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_compute_moving_average(self):
        # For a known input, check that the moving average is computed correctly.
        data = np.array([1, 2, 3, 4, 5])
        result = CEBRAUtils.compute_moving_average(data, window_size=2)
        expected = np.array([1.5, 2.5, 3.5, 4.5])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_get_var_over_lap(self):
        # Test that the lap calculation and sorting work as expected.
        var = np.array([30, 10, 20])
        true_angle = np.array([2*np.pi, np.pi, 0])  # lap numbers: 1, 0.5, 0
        lap_number, sorted_var, sorted_lap = CEBRAUtils.get_var_over_lap(var, true_angle)
        expected_lap = np.array([0, 0.5, 1])
        # sorted_var should match the var corresponding to sorted lap numbers.
        # For true_angle [0, np.pi, 2*np.pi] sorted, var becomes [20, 10, 30].
        expected_sorted_var = np.array([20, 10, 30])
        np.testing.assert_allclose(sorted_lap, expected_lap, rtol=1e-5)
        np.testing.assert_allclose(sorted_var, expected_sorted_var, rtol=1e-5)

    def test_low_pass_filter(self):
        # Generate a noisy sine wave and ensure the filtered output is smoother.
        t = np.linspace(0, 2*np.pi, 100)
        x = np.sin(t) + 0.5*np.random.randn(100)
        filtered = CEBRAUtils.low_pass_filter(x, cutoff_frequency=0.1, filter_order=3, fs=1)
        # Check the shape is unchanged and the standard deviation is reduced.
        self.assertEqual(filtered.shape, x.shape)
        self.assertLess(np.std(filtered), np.std(x))

    def test_nt_TDA_mask(self):
        # Create a small dataset with an obvious outlier.
        data = np.random.rand(9, 3)  # 9 normal points

        # Append an outlier far from the others
        outlier = np.array([[100, 100, 100]])  # A point far away in space
        data = np.vstack((data, outlier))  # Add the outlier to the dataset

        # Run nt_TDA_mask
        mask = CEBRAUtils.nt_TDA_mask(data, verbose=False)

        # Ensure the mask length is correct
        self.assertEqual(len(mask), 10)  

        # Check that at least one outlier is detected
        self.assertFalse(mask[-1], "The outlier was not detected!")  # Last point should be an outlier

    def test_mean_dist_to_spline(self):
        # Define a simple principal curve as a straight line.
        principal_curve = np.array([[0, 0, 0],
                                    [0, 0, 1],
                                    [0, 0, 2]])
        # Create embeddings that are close to, but not exactly on, the principal curve.
        # These points have a small perpendicular offset from the line.
        embeddings = np.array([[0.1, 0.0, 0.0],
                            [0.5, 0.6, 0.5],
                            [1.5, 1.4, 1.6],
                            [2.1, 2.0, 2.2]])
        
        mean_dist, distances = CEBRAUtils.mean_dist_to_spline(embeddings, principal_curve)
        self.assertEqual(distances[0],0.1,"Distance wrong from spline")
        # Check that the mean distance is greater than zero.
        self.assertGreater(mean_dist, 0, "Mean distance should be greater than 0 for points off the principal curve.")
        
        # Optionally, also check that the distances array has the same length as the number of embeddings.
        self.assertEqual(len(distances), embeddings.shape[0])

    def test_compute_betti_numbers(self):
        # Create dummy persistence diagrams.
        # For H0, use two features with lifespans 1 and 0.7; for H1, an empty array.
        H0 = np.array([[0, 1], [0.2, 0.9]])
        H1 = np.empty((0, 2))
        betti_0, betti_1 = CEBRAUtils.compute_betti_numbers(H0, H1, lifespan_fraction=0.5)
        # Lifespan of first feature: 1, second: 0.7, threshold=0.5 so both count.
        self.assertEqual(betti_0, 2)
        self.assertEqual(betti_1, 0)

    # Optionally, add tests for plotting functions to simply ensure that no exceptions are raised.
    def test_plot_initial_knots(self):
        # Create dummy data for a 3D scatter plot.
        data_points = np.random.rand(10, 3)
        init_knots = np.random.rand(3, 3)
        # Call the function (this should create and then show/close a figure).
        try:
            CEBRAUtils.plot_initial_knots(data_points, init_knots, session_idx=1, session=type("dummy", (), {"rat": "Test", "day": "Day1", "epoch": "E1"}))
        except Exception as e:
            self.fail(f"plot_initial_knots raised an exception {e}")

if __name__ == '__main__':
    unittest.main()
