from ai_coach import AICoach
import yaml


def main():
    with open("../configs/ai_coach.yaml", "r") as stream:
        args = yaml.safe_load(stream)
    
    aic = AICoach(args)
    aic.start_resolver()

if __name__ == "__main__":
    main()