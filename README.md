# Topological Hough Transform Project

This project implements a **Topological Hough Transform** for detecting lines in images, alongside a baseline Hough Transform for comparison. It includes tools for generating synthetic data, visualizing results, and evaluating performance using confusion matrices.

## Features

- **Topological Hough Transform**: A novel approach to line detection with persistence diagrams.
- **Baseline Hough Transform**: Standard OpenCV-based line detection for comparison.
- **Synthetic Data Generation**: Create noisy lines for testing and evaluation.
- **Visualization**: Plot persistence diagrams, Hough spaces, and detected lines.
- **Evaluation**: Generate confusion matrices to compare detection methods.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/martinuray/TopologicalHoughTransform.git
cd TopologicalHoughTransformation
pip install -r requirements.txt
```

## Using the library module: Examples

### Generating a Persistence Diagram
```python
from src.topological_hough_transform import TopologicalHoughTransform
from utils.plotting import plot_persistence_diagram

image = ...  # Load or generate an image
hough_transformer = TopologicalHoughTransform(image, value_threshold=150, pers_limit=120)
plot_persistence_diagram(hough_transformer)
```

### Usage of Experiments
The experiments shall also be used as a reference on how to use the library
module. The experiments are located in the `experiments` folder and can be run
directly. Run the main experiments, as described in the initial publication [1]
using:
```bash
python experiments/experiment_1.py --output-directory results --log-level INFO
```

#### Arguments
- `--output-directory`: Directory to save results.
- `--log-level`: Logging level (e.g., `INFO`, `DEBUG`).
- Additional arguments can be found in the `create_parser` function in `utils/parser.py` and are listed using
```bash
python experiments/experiment_1.py --help
```

The same applies also to the other experiments.

#### Example
```bash
python experiments/experiment_1.py --output-directory results --noise-levels 1 2 3 --num-sim-rounds 10
```

To reproduce the experiment 1 from the paper, use the following arguments:
```bash
python experiments/experiment_1.py --num-sim-rounds 100 --noise-levels 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```
Experiment 2 can be reproduced by just running the corresponding python script:
```bash
python experiments/experiment_2.py
```
Experiment 3 is split into two parts. First, you run the stability experiment that generates a csv file:
```bash
python experiments/experiment_3_stability.py
```
Afterwards, you can run the plotting script.
```bash
python experiments/experiment_3_plotting.py
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

```bibtex
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
