"""Getting params from the command line."""

from argparse import ArgumentParser

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = ArgumentParser(description="Resolve Badminton Videos")

    parser.add_argument(
        "--yaml_path",
        type=str,
        default="../configs/ai_coach.yaml",
        help="Path to AI Coach setting yaml."
    )
    
    return parser.parse_args()