from ai_coach import AICoach

def main():
    args = {
        'sacnn_path': 'models/sacnn.pt',
        'saqueue length': 5,
    }
    video_directory = '../videos'
    aic = AICoach(args, video_directory)
    aic.start_resolver()

if __name__ == "__main__":
    main()