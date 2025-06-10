"""
test_transform.py
Author: J. Ferner, S. Huber, S. Messineo, A. Pop, M. Uray
Date: June 2025
Description: Unit tests for the transformation functions between
                slope-intercept and polar coordinates (rho, theta) for lines,
                and for converting lines to points.
License: MIT
"""
import unittest

from topologicalhoughtransform.utils.transform import \
    slope_intercept_to_rho_theta, rho_theta_to_slope_intercept, line_to_pts


class TestMathUtils(unittest.TestCase):
    """Unit tests for the transformation functions."""

    def test_slope_intercept_to_rho_theta_horizontal_line(self):
        """Test conversion of horizontal lines to rho, theta."""
        line = (0, 5)  # Horizontal line: y = 5
        rho, theta = slope_intercept_to_rho_theta(line)
        self.assertAlmostEqual(rho, 5, places=5)
        self.assertAlmostEqual(theta, 90, places=5)

        line = (0, -5)  # Horizontal line: y = -5
        rho, theta = slope_intercept_to_rho_theta(line)
        self.assertAlmostEqual(rho, -5, places=5)
        self.assertAlmostEqual(theta, 90, places=5)

    def test_slope_intercept_to_rho_theta_vertical_line(self):
        """Test conversion of vertical lines to rho, theta."""
        line = (float('inf'), -3)  # Vertical line: x = -3
        rho, theta = slope_intercept_to_rho_theta(line)
        self.assertAlmostEqual(rho, 3, places=5)
        self.assertEqual(theta, 0)

    def test_slope_intercept_to_rho_theta_diagonal_line(self):
        """Test conversion of diagonal lines to rho, theta."""
        line = (1, 0)  # Diagonal line: y = x
        rho, theta = slope_intercept_to_rho_theta(line)
        self.assertAlmostEqual(rho, 0, places=5)
        self.assertAlmostEqual(theta, 135, places=5)

    def test_rho_theta_to_slope_intercept_horizontal_line(self):
        """Test conversion of rho, theta to slope-intercept for horizontal
        lines."""
        line = (5, 90)  # Horizontal line: y = 5
        k, d = rho_theta_to_slope_intercept(line)
        self.assertAlmostEqual(k, 0, places=5)
        self.assertAlmostEqual(d, 5, places=5)

    def test_rho_theta_to_slope_intercept_vertical_line(self):
        """Test conversion of rho, theta to slope-intercept for vertical
        lines."""
        line = (3, 0)  # Vertical line: x = 3
        k, d = rho_theta_to_slope_intercept(line)
        self.assertEqual(k, float('inf'))
        self.assertAlmostEqual(d, -3, places=5)

    def test_rho_theta_to_slope_intercept_diagonal_line(self):
        """Test conversion of rho, theta to slope-intercept for diagonal
        lines."""
        line = (0, 135)  # Diagonal line: y = x
        k, d = rho_theta_to_slope_intercept(line)
        self.assertAlmostEqual(k, 1, places=5)
        self.assertAlmostEqual(d, 0, places=5)

    def test_line_to_pts_horizontal_line(self):
        """Test conversion of horizontal lines to points."""
        line = (5, 90)  # Horizontal line: y = 5
        pt1, pt2 = line_to_pts(line)
        self.assertEqual(pt1[1], 5)
        self.assertEqual(pt2[1], 5)
        self.assertEqual(pt1[0], -pt2[0])

    def test_line_to_pts_vertical_line(self):
        """Test conversion of vertical lines to points."""
        line = (3, 0)  # Vertical line: x = 3
        pt1, pt2 = line_to_pts(line)
        self.assertEqual(pt1[0], 3)
        self.assertEqual(pt2[0], 3)
        self.assertEqual(pt1[1], -pt2[1])

    def test_line_to_pts_diagonal_line(self):
        """Test conversion of diagonal lines to points."""
        line = (0, 135)  # Diagonal line: y = x
        pt1, pt2 = line_to_pts(line)
        self.assertNotEqual(pt1[0], pt2[0])
        self.assertNotEqual(pt1[1], pt2[1])

    def test_conversion_consistency(self):
        """Test that converting to polar and back gives the original line."""
        original_line = (1, 2)  # y = x + 2
        rho, theta = slope_intercept_to_rho_theta(original_line)
        k, d = rho_theta_to_slope_intercept((rho, theta))
        self.assertAlmostEqual(original_line[0], k, places=5)
        self.assertAlmostEqual(original_line[1], d, places=5)


if __name__ == "__main__":
    unittest.main()
