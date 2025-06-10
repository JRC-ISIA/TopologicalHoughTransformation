import unittest
import numpy as np
from topologicalhoughtransform.topological_hough_transform import TopologicalHoughTransform

class TestTopologicalHoughTransform(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.image = np.zeros((100, 100), dtype=np.uint8)
        self.transform = TopologicalHoughTransform(self.image)

    def test_initialization(self):
        """Test if the class initializes correctly."""
        self.assertIsInstance(self.transform, TopologicalHoughTransform)

    def test_invalid_image_type(self):
        """Test initialization with an invalid image type."""
        with self.assertRaises(ValueError):
            TopologicalHoughTransform("invalid_image")

    def test_detect_lines_empty_image(self):
        """Test line detection on an empty image."""
        lines = self.transform.get_lines()
        self.assertEqual(len(lines), 0)

    def test_detect_lines_with_noise(self):
        """Test line detection on a noisy image."""
        noisy_image = self.image + np.random.randint(0, 255, self.image.shape, dtype=np.uint8)
        transform = TopologicalHoughTransform(noisy_image)
        lines = transform.get_lines()
        self.assertIsInstance(lines, list)

    def test_detect_lines_with_single_line(self):
        """Test line detection on an image with a single line."""
        self.image[50, :] = 255  # Add a horizontal line
        transform = TopologicalHoughTransform(self.image)
        lines = transform.get_lines()
        self.assertGreater(len(lines), 0)

    def test_detect_lines_with_multiple_lines(self):
        """Test line detection on an image with multiple lines."""
        self.image[50, :] = 255  # Add a horizontal line
        self.image[:, 50] = 255  # Add a vertical line
        transform = TopologicalHoughTransform(self.image)
        lines = transform.get_lines()
        self.assertGreaterEqual(len(lines), 2)

    def test_detect_lines_with_curved_lines(self):
        """Test line detection on an image with curved lines."""
        for i in range(100):
            self.image[i, i] = 255  # Add a diagonal line
        transform = TopologicalHoughTransform(self.image)
        lines = transform.get_lines()
        self.assertGreater(len(lines), 0)

    def test_detect_lines_with_noisy_curved_lines(self):
        """Test line detection on an image with noisy curved lines."""
        for i in range(100):
            self.image[i, i] = 255  # Add a diagonal line
        noisy_image = self.image + np.random.randint(0, 50, self.image.shape, dtype=np.uint8)
        transform = TopologicalHoughTransform(noisy_image)
        lines = transform.get_lines()
        self.assertGreater(len(lines), 0)

if __name__ == "__main__":
    unittest.main()