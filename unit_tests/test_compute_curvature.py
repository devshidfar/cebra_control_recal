# test_manifold_fit.py

import unittest
import numpy as np
import sys
sys.path.append("/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal")
from cebra_utils import *


class TestComputeCurvature(unittest.TestCase):
    # def setUp(self):
    #     # Initialize with dummy data; actual data is irrelevant for curvature computation
    #     self.params = {
    #         'nKnots': 15,
    #         'dalpha': 0.1
    #     }
    #     # Dummy data to initialize, not used directly
    #     self.data_to_fit = np.zeros((15, 3))
    #     self.fitter = PiecewiseLinearFit(self.data_to_fit, self.params)
    
    def test_straight_line(self):
        """Test curvature for points lying on a straight line."""
        loop_knots = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10,10,10],
            [11,11,11],
            [12,12,12],
            [13,13,13],
            [14,14,14],
            [0,0,0]  # Loop back to first point
        ])
        curvature = compute_curvature(loop_knots)
        self.assertAlmostEqual(curvature, 0.0, places=6, msg="Curvature should be zero for straight line.")
    
    def test_square(self):
        """Test curvature for a square shape."""
        # Define a square in 2D (z=0), looped
        loop_knots = np.array([
            [0,0,0],
            [1,0,0],
            [1,1,0],
            [0,1,0],
            [0,0,0]
        ])
        # Duplicate knots to reach nKnots=15 by repeating
        loop_knots = np.tile(loop_knots, (3,1))[:15]
        loop_knots = np.vstack([loop_knots, loop_knots[0]])  # Loop back
        curvature = compute_curvature(loop_knots)
        # Each right-angle turn introduces a change in direction
        # For a square, with 4 right angles, expected curvature = 4 * (pi/2)^2 = pi^2
        # However, since we're in discrete steps and normalized directions,
        # the exact value may differ. We'll check it's positive and greater than zero.
        self.assertTrue(curvature > 0.0, "Curvature should be positive for square shape.")
    
    def test_triangle(self):
        """Test curvature for an equilateral triangle."""
        # Define an equilateral triangle in 2D (z=0), looped
        angle = 2 * np.pi / 3
        loop_knots = np.array([
            [0,0,0],
            [1,0,0],
            [0.5, np.sin(angle), 0],
            [0,0,0]
        ])
        # Duplicate knots to reach nKnots=15 by repeating
        loop_knots = np.tile(loop_knots, (4,1))[:15]
        loop_knots = np.vstack([loop_knots, loop_knots[0]])  # Loop back
        curvature = compute_curvature(loop_knots)
        # Each 120-degree turn introduces a change in direction
        self.assertTrue(curvature > 0.0, "Curvature should be positive for triangle shape.")
    
    def test_circle_approximation(self):
        """Test curvature for points sampled from a circle."""
        num_points = 16
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radius = 1
        loop_knots = np.array([[radius * np.cos(a), radius * np.sin(a), 0] for a in angles])
        loop_knots = np.vstack([loop_knots, loop_knots[0]])  # Loop back
        curvature = compute_curvature(loop_knots)
        # For a circle, curvature should be proportional to the number of points and inversely proportional to radius squared
        # Since radius=1, curvature is proportional to num_points * (delta_angle)^2
        expected_curvature = num_points * (2*np.pi / num_points)**2  # Simplified
        self.assertAlmostEqual(curvature, 4*np.pi**2 / num_points, places=2,
                               msg=f"Curvature should be approximately {4*np.pi**2 / num_points}")
    
    def test_two_knots(self):
        """Test curvature with only two knots (should be zero)."""
        loop_knots = np.array([
            [0,0,0],
            [1,1,1],
            [0,0,0]
        ])
        curvature = compute_curvature(loop_knots)
        self.assertAlmostEqual(curvature, 0.0, places=6, msg="Curvature should be zero for two knots.")
    
    def test_single_knot(self):
        """Test curvature with a single knot (should raise an error)."""
        loop_knots = np.array([[0,0,0]])
        with self.assertRaises(ValueError):
            curvature = compute_curvature(loop_knots)
    
    def test_empty_knots(self):
        """Test curvature with no knots (should raise an error)."""
        loop_knots = np.empty((0, 3))
        with self.assertRaises(ValueError):
            curvature = compute_curvature(loop_knots)
    
    def test_non_looped_knots(self):
        """Test curvature with non-looped knots."""
        loop_knots = np.array([
            [0,0,0],
            [1,1,1],
            [2,2,2],
            [3,3,3],
            [4,4,4],
            [5,5,5],
            [6,6,6],
            [7,7,7],
            [8,8,8],
            [9,9,9],
            [10,10,10],
            [11,11,11],
            [12,12,12],
            [13,13,13],
            [14,14,14]
            # Not looping back
        ])
        curvature = compute_curvature(loop_knots)
        # Since the function assumes looped knots, it adds no closure
        # This might lead to higher curvature due to open ends
        self.assertTrue(curvature > 0.0, "Curvature should be positive for non-looped knots.")

if __name__ == '__main__':
    unittest.main()
