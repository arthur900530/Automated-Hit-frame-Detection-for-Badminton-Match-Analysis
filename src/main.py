from ai_coach import AICoach

def main():
    args = {
        'sacnn_path': 'src/models/sacnn.pt',
        'court_kpRCNN_path': 'src/models/court_kpRCNN.pth',
        'saqueue length': 5,
        'video_directory': 'videos',
    }
    
    aic = AICoach(args)
    aic.start_resolver()

if __name__ == "__main__":
    main()