import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description="Parser for the experiments for the topological variant "
                    "of the hough transform.")

    # Group 1: Input configuration
    experiment_group = parser.add_argument_group("Experiment Configuration")
    experiment_group.add_argument("--num-sim-rounds", type=int,
                                  default=10,
                                  help="Number of simulation rounds.")
    # Add an argument for a list of integers with a default value
    experiment_group.add_argument(
        "--noise-levels", type=int, nargs="+",
        default=[8, 9], help="List of noise levels to be used."
    )

    experiment_group.add_argument(
        "--line-1-intercept", type=float, default=0.,
        help="Intercept for first line."
    )
    experiment_group.add_argument(
        "--line-1-slope", type=float, default=1.,
        help="Slope for first line.")
    experiment_group.add_argument(
        "--line-2-intercept", type=float, default=0.,
        help="Intercept for second line."
    )
    experiment_group.add_argument(
        "--line-2-slope", type=float, default=1.,
        help="Slope for second line.")
    experiment_group.add_argument(
        "--n-point-line-1", type=int, default=400,
        help="Number of points for first line."
    )
    experiment_group.add_argument(
        "--n-point-line-2", type=int, default=350,
        help="Number of points for second line."
    )

    # Group 2: Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output-directory", type=str,
                              default='out',
                              help="Path to store the output artifacts.")

    # Group 3: Misc
    misc_group = parser.add_argument_group("Logging Configuration")
    misc_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)."
    )

    # Group 4: OpenCV method
    cv_group = parser.add_argument_group("OpenCV Configuration")
    cv_group.add_argument(
        "--opencv-threshold",
        type=int,
        default=45,
        help="Threshold for OpenCV line detection (default: 45)."
    )

    return parser
