from ai_coach import AICoach
from param_parser import parameter_parser
import yaml

def main():
    args = parameter_parser()

    with open(args.yaml_path, "r") as stream:
        args = yaml.safe_load(stream)
    
    aic = AICoach(args)
    aic.start_resolver()

if __name__ == "__main__":
    main()