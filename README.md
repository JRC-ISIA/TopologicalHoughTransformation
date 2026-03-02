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
```

## Using the repository

The experiments for the iDSC'25 paper 
["Persistence-Based Hough Transform for Line Detection"](https://arxiv.org/abs/2504.16114) 
can be found in the `experiments_idsc25` folder. 
Each experiment is organized in a separate python script.

The code and experiments for the EuroCG'26 paper 
["Topologically Stable Hough Transform"]()
will follow soon!


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

@inproceedings{huber2026,
  title={Topologically Stable Hough Transform},
  author={Stefan Huber and Krist{\'o}f Husz{\'a}r and Michael Kerber and Martin Uray},
  conference={26th European Workshop on Computational Geometry},
  year={2026},
}
```

## Acknowledgments
he financial support by the Austrian Federal Ministry of Economy, Energy and 
Tourism, the National Foundation for Research, Technology and Development and 
the Christian Doppler Research Association is gratefully acknowledged.

We thank [Lukas Lürzer](https://orcid.org/0009-0000-5953-1381) for the help 
on preparing the code for publication.

## Contact

For questions or feedback, please contact [Martin Uray](martin.uray@fh-salzburg.ac.at).
