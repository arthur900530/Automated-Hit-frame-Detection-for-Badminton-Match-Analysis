from ai_coach import AICoach

def main():
    args = {
        'sacnn_path': 'src/models/sacnn.pt',
        'court_kpRCNN_path': 'src/models/court_kpRCNN.pth',
        'saqueue length': 5,
        'video_directory': 'videos',
        'kpRCNN_path': 'src/models/kpRCNN.pth',
        'video_save_path': 'outputs/videos',
        'joint_save_path': 'outputs/joints',
        'rally_save_path': 'outputs/rallies',
        'opt_path': 'src/models/OPT_16_head_dp.pt',
        'scaler_path': 'src/models/scaler_new.pickle',
    }
    
    aic = AICoach(args)
    aic.start_resolver()

if __name__ == "__main__":
    main()