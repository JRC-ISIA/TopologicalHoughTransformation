# Topological Hough Transform Project

This project implements a **Topological Hough Transform** for detecting lines in images, alongside a baseline Hough Transform for comparison. It includes tools for generating synthetic data, visualizing results, and evaluating performance using confusion matrices.

## Features

- **Topological Hough Transform**: A novel approach to line detection with persistence diagrams.
- **Baseline Hough Transform**: Standard OpenCV-based line detection for comparison.
- **Synthetic Data Generation**: Create noisy lines for testing and evaluation.
- **Visualization**: Plot persistence diagrams, Hough spaces, and detected lines.
- **Evaluation**: Generate confusion matrices to compare detection methods.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments
Run the main experiment script:
```bash
python experiments/experiment_1.py --output_directory results --log_level INFO
```

### Arguments
- `--output_directory`: Directory to save results.
- `--log_level`: Logging level (e.g., `INFO`, `DEBUG`).
- Additional arguments can be found in the `create_parser` function in `utils/parser.py`.

### Example
```bash
python experiments/experiment_1.py --output_directory results --noise_levels 0.1 0.2 0.3 --num_sim_rounds 10
```

### Visualizing Results
- Persistence diagrams and Hough spaces are saved in the `results/tmp` directory.
- Use the `utils/plotting.py` module to customize visualizations.

## Project Structure

```
.
├── experiments/                # Experiment scripts
├── topologicalhoughtransform/  # Core implementation
│   ├── utils/                  # Utility functions
│   ├── TopologicalHoughTransform.py  # Main class
├── utils/                      # Additional utilities
└── README.md                   # Project documentation
```

## Key Modules

- **`TopologicalHoughTransform`**: Implements the topological Hough transform.
- **`utils/plotting.py`**: Functions for visualizing results.
- **`utils/test_data_generator.py`**: Generate synthetic data for testing.
- **`utils/parser.py`**: Command-line argument parser.

## Examples

### Generating a Persistence Diagram
```python
from topologicalhoughtransform.TopologicalHoughTransform import TopologicalHoughTransform
from utils.plotting import plot_persistence_diagram

image = ...  # Load or generate an image
hough_transformer = TopologicalHoughTransform(image, value_threshold=150, pers_limit=120)
plot_persistence_diagram(hough_transformer)
```

### Visualizing Hough Space
```python
from unsure.utils import hough_line, show_hough_line

accumulator, thetas, rhos = hough_line(image)
show_hough_line(image, accumulator, thetas, rhos)
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Cite
In case you think this project is useful for your work, please cite it as follows:

```
@online{ferner2025,
  title = {Persistence-Based {{Hough Transform}} for {{Line Detection}}},
  author = {Ferner, Johannes and Huber, Stefan and Messineo, Saverio and Pop, Angel and Uray, Martin},
  date = {2025-05-22},
  eprint = {2504.16114},
  eprinttype = {arXiv},
  eprintclass = {cs},
  doi = {10.48550/arXiv.2504.16114},
  url = {http://arxiv.org/abs/2504.16114},
}

```


## Contact

For questions or feedback, please contact [Martin Uray](martin.uray@fh-salzburg.ac.at).
