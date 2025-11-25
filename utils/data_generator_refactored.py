"""
data_generator_refactored.py
Author: GitHub Copilot (Refactored)
Date: November 2025
Description: Refactored synthetic test data generator with improved architecture
License: MIT
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Union
import random

import numpy as np
from PIL import Image


@dataclass
class ImageConfig:
    """Configuration for image generation."""
    width: int = 256
    height: int = 256
    background_color: int = 0
    foreground_color: int = 255


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    level: float = 5.0
    distribution: str = "normal"  # "normal", "uniform"
    orthogonal: bool = True  # Apply noise orthogonally to the line


class ShapeType(Enum):
    """Enumeration of available shape types."""
    LINE = "line"
    VERTICAL_LINE = "vertical_line"
    HOUGH_LINE = "hough_line"
    BENT_LINE = "bent_line"
    CIRCLE = "circle"
    ARC = "arc"


class AbstractShapeGenerator(ABC):
    """Abstract base class for shape generators."""
    
    def __init__(self, image_config: ImageConfig):
        self.image_config = image_config
    
    @abstractmethod
    def generate_points(self, **kwargs) -> List[Tuple[int, int]]:
        """Generate points for the shape."""
        pass
    
    def _point_in_bounds(self, x: int, y: int) -> bool:
        """Check if point is within image bounds."""
        return (0 <= x < self.image_config.width and 
                0 <= y < self.image_config.height)
    
    def _apply_noise(self, points: np.ndarray, noise_config: NoiseConfig, 
                    slope: Optional[float] = None) -> np.ndarray:
        """Apply noise to points."""
        if noise_config.level == 0:
            return points
            
        num_points = len(points)
        
        if noise_config.orthogonal and slope is not None:
            # Orthogonal noise for lines
            theta = np.arctan(slope)
            dx = np.sin(theta)
            dy = -np.cos(theta)
            
            if noise_config.distribution == "uniform":
                noise_mag = np.random.uniform(-noise_config.level, 
                                            noise_config.level, num_points)
            else:
                noise_mag = np.random.normal(0, noise_config.level, num_points)
            
            noise_x = noise_mag * dx
            noise_y = noise_mag * dy
        else:
            # Simple random noise
            if noise_config.distribution == "uniform":
                noise_x = np.random.uniform(-noise_config.level, 
                                          noise_config.level, num_points)
                noise_y = np.random.uniform(-noise_config.level, 
                                          noise_config.level, num_points)
            else:
                noise_x = np.random.normal(0, noise_config.level, num_points)
                noise_y = np.random.normal(0, noise_config.level, num_points)
        
        noisy_points = points + np.column_stack([noise_x, noise_y])
        return np.round(noisy_points).astype(np.int32)
    
    def _filter_valid_points(self, points: np.ndarray) -> np.ndarray:
        """Filter out points that are outside image bounds."""
        valid_mask = np.array([
            self._point_in_bounds(x, y) for x, y in points
        ])
        return points[valid_mask]


class LineGenerator(AbstractShapeGenerator):
    """Generator for straight lines."""
    
    def generate_points(self, slope: float = 1.0, intercept: float = 0.0,
                       num_points: int = 100, 
                       noise_config: Optional[NoiseConfig] = None) -> List[Tuple[int, int]]:
        """Generate points for a straight line."""
        noise_config = noise_config or NoiseConfig()
        
        # Generate x coordinates
        if noise_config.level == 0:
            # For zero noise, use evenly distributed points
            x_coords = np.linspace(0, self.image_config.width - 1, 
                                 min(num_points, self.image_config.width))
            np.random.shuffle(x_coords)
            x_coords = x_coords[:num_points]
        else:
            x_coords = np.random.randint(0, self.image_config.width, num_points)
        
        # Calculate y coordinates
        y_coords = slope * x_coords + intercept
        
        points = np.column_stack([x_coords, y_coords])
        
        # Apply noise
        if noise_config.level > 0:
            points = self._apply_noise(points, noise_config, slope)
        
        # Filter valid points and remove duplicates
        points = self._filter_valid_points(points)
        points = np.unique(points, axis=0)
        
        # Ensure we have enough points
        while len(points) < num_points:
            additional_points = self._generate_additional_points(
                slope, intercept, num_points - len(points), noise_config)
            points = np.vstack([points, additional_points])
            points = self._filter_valid_points(points)
            points = np.unique(points, axis=0)
        
        return points[:num_points].tolist()
    
    def _generate_additional_points(self, slope: float, intercept: float,
                                  num_needed: int, 
                                  noise_config: NoiseConfig) -> np.ndarray:
        """Generate additional points to meet the required count."""
        x_coords = np.random.randint(0, self.image_config.width, num_needed * 2)
        y_coords = slope * x_coords + intercept
        points = np.column_stack([x_coords, y_coords])
        
        if noise_config.level > 0:
            points = self._apply_noise(points, noise_config, slope)
        
        return self._filter_valid_points(points)


class HoughLineGenerator(AbstractShapeGenerator):
    """Generator for lines using Hough parameters."""
    
    def generate_points(self, rho: float, theta: float, num_points: int = 100,
                       noise_config: Optional[NoiseConfig] = None,
                       origin_shift: bool = True) -> List[Tuple[int, int]]:
        """Generate points for a line using Hough parameters."""
        noise_config = noise_config or NoiseConfig()
        points = []
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Handle vertical lines
        if np.isclose(sin_theta, 0, atol=1e-9):
            x = self.image_config.width // 2 if origin_shift else int(rho)
            
            for _ in range(num_points):
                y = random.randint(0, self.image_config.height - 1)
                noisy_x = x + random.randint(-int(noise_config.level), 
                                           int(noise_config.level))
                if self._point_in_bounds(noisy_x, y):
                    points.append((noisy_x, y))
        else:
            # Normal line processing
            if origin_shift:
                x0, y0 = self.image_config.width // 2, self.image_config.height // 2
            else:
                x0, y0 = rho * cos_theta, rho * sin_theta
            
            t_max = max(self.image_config.width, self.image_config.height)
            
            attempts = 0
            while len(points) < num_points and attempts < num_points * 10:
                t = random.uniform(-t_max, t_max)
                x = int(x0 + t * -sin_theta)
                y = int(y0 + t * cos_theta)
                
                # Add noise
                if noise_config.level > 0:
                    noise_mag = random.randint(-int(noise_config.level), 
                                             int(noise_config.level))
                    x += int(noise_mag * -sin_theta)
                    y += int(noise_mag * -cos_theta)
                
                if self._point_in_bounds(x, y):
                    points.append((x, y))
                
                attempts += 1
        
        return points


class CircleGenerator(AbstractShapeGenerator):
    """Generator for circles."""
    
    def generate_points(self, radius: Optional[float] = None, 
                       center: Optional[Tuple[float, float]] = None,
                       num_points: int = 100,
                       noise_config: Optional[NoiseConfig] = None) -> List[Tuple[int, int]]:
        """Generate points for a circle."""
        noise_config = noise_config or NoiseConfig()
        
        if radius is None:
            radius = min(self.image_config.width, self.image_config.height) // 2 - 45
        
        if center is None:
            center = (self.image_config.width // 2, self.image_config.height // 2)
        
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x_coords = center[0] + radius * np.cos(angles)
        y_coords = center[1] + radius * np.sin(angles)
        
        points = np.column_stack([x_coords, y_coords])
        
        # Apply noise
        if noise_config.level > 0:
            noise_x = np.random.randint(-int(noise_config.level), 
                                      int(noise_config.level) + 1, num_points)
            noise_y = np.random.randint(-int(noise_config.level), 
                                      int(noise_config.level) + 1, num_points)
            points += np.column_stack([noise_x, noise_y])
        
        points = np.round(points).astype(np.int32)
        return self._filter_valid_points(points).tolist()


class SyntheticDataGenerator:
    """Main class for generating synthetic data."""
    
    def __init__(self, image_config: Optional[ImageConfig] = None):
        self.image_config = image_config or ImageConfig()
        self._generators = {
            ShapeType.LINE: LineGenerator(self.image_config),
            ShapeType.HOUGH_LINE: HoughLineGenerator(self.image_config),
            ShapeType.CIRCLE: CircleGenerator(self.image_config),
        }
    
    def generate_shape(self, shape_type: ShapeType, **kwargs) -> List[Tuple[int, int]]:
        """Generate points for a specific shape type."""
        if shape_type not in self._generators:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        
        return self._generators[shape_type].generate_points(**kwargs)
    
    def generate_line(self, slope: float = 1.0, intercept: float = 0.0,
                     num_points: int = 100, noise_level: float = 5.0,
                     noise_distribution: str = "normal") -> List[Tuple[int, int]]:
        """Generate a straight line (convenience method)."""
        noise_config = NoiseConfig(level=noise_level, distribution=noise_distribution)
        return self.generate_shape(
            ShapeType.LINE, 
            slope=slope, 
            intercept=intercept,
            num_points=num_points,
            noise_config=noise_config
        )
    
    def generate_hough_line(self, rho: float, theta: float, num_points: int = 100,
                           noise_level: float = 0.0, origin_shift: bool = True) -> List[Tuple[int, int]]:
        """Generate a line using Hough parameters (convenience method)."""
        noise_config = NoiseConfig(level=noise_level)
        return self.generate_shape(
            ShapeType.HOUGH_LINE,
            rho=rho,
            theta=theta,
            num_points=num_points,
            noise_config=noise_config,
            origin_shift=origin_shift
        )
    
    def generate_circle(self, radius: Optional[float] = None,
                       center: Optional[Tuple[float, float]] = None,
                       num_points: int = 100, noise_level: float = 5.0) -> List[Tuple[int, int]]:
        """Generate a circle (convenience method)."""
        noise_config = NoiseConfig(level=noise_level)
        return self.generate_shape(
            ShapeType.CIRCLE,
            radius=radius,
            center=center,
            num_points=num_points,
            noise_config=noise_config
        )
    
    def generate_noise_points(self, num_points: int = 100) -> List[Tuple[int, int]]:
        """Generate random noise points."""
        points = np.random.randint(
            [0, 0], 
            [self.image_config.width, self.image_config.height], 
            size=(num_points, 2)
        )
        return points.tolist()
    
    def create_image(self, coordinates: List[Tuple[int, int]]) -> Image.Image:
        """Create a PIL Image from coordinates."""
        image = Image.new('L', (self.image_config.width, self.image_config.height), 
                         self.image_config.background_color)
        
        for x, y in coordinates:
            if 0 <= x < self.image_config.width and 0 <= y < self.image_config.height:
                image.putpixel((x, y), self.image_config.foreground_color)
        
        return image


# Backward compatibility functions
def generate_line(args, noise_lvl=5, slope=1, intercept=0, num_points=100, equaldist=False):
    """Backward compatibility function for generate_line."""
    config = ImageConfig(width=args.image_width, height=args.image_height)
    generator = SyntheticDataGenerator(config)
    
    distribution = "uniform" if equaldist else "normal"
    return generator.generate_line(
        slope=slope, 
        intercept=intercept, 
        num_points=num_points,
        noise_level=noise_lvl,
        noise_distribution=distribution
    )


def generate_image(coordinates, args):
    """Backward compatibility function for generate_image."""
    config = ImageConfig(width=args.image_width, height=args.image_height)
    generator = SyntheticDataGenerator(config)
    return generator.create_image(coordinates)


def generate_hough_line(rho, theta, args, noise_lvl=0, num_points=100, origin_shift=True):
    """Backward compatibility function for generate_hough_line."""
    config = ImageConfig(width=args.image_width, height=args.image_height)
    generator = SyntheticDataGenerator(config)
    return generator.generate_hough_line(
        rho=rho, 
        theta=theta, 
        num_points=num_points,
        noise_level=noise_lvl, 
        origin_shift=origin_shift
    )


def generate_circle(args, noise=5, num_points=100):
    """Backward compatibility function for generate_circle."""
    config = ImageConfig(width=args.image_width, height=args.image_height)
    generator = SyntheticDataGenerator(config)
    return generator.generate_circle(num_points=num_points, noise_level=noise)


def point_in_image(x, y, args):
    """Backward compatibility function for point_in_image."""
    return 0 <= x < args.image_width and 0 <= y < args.image_height


def generate_noise(args, num_points=100):
    """Backward compatibility function for generate_noise."""
    config = ImageConfig(width=args.image_width, height=args.image_height)
    generator = SyntheticDataGenerator(config)
    return generator.generate_noise_points(num_points)


# Utility functions
def rgb2gray(rgb):
    """Convert RGB image to grayscale using the luminosity method."""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def _round_np_to_int(arr):
    """Round numpy array to integers."""
    return np.round(arr).astype(np.int32)
