# Transformer on Shuttlecock Flying Direction Prediction for Hit-frame Detection
This repository stores the code that aims to advance sport science in badminton, providing a systematic approach to generate stroke & player movement sequences automatically from game videos using modern AI techniques. A deep evaluation on BWF records can then be conducted.
CNN and Keypoint R-CNN models are applied to extract frames, and collect court and player information. Moreover, a Transformer model to transform player joint sequences to shuttlecock direction sequences.
## Environment Requirements
```
opencv-python == 4.7.0.72
python == 3.8
scikit-learn == 1.0.2
torch == 2.0.1
torchvision == 0.15.2
yaml
```
## YAML Parameters
```
- model:
  - sacnn_path: Path to SA-CNN's weight                               Default is './models/weights/sacnn.pt'.
  - court_kpRCNN_path: Path to Court Keypoint-RCNN's weight           Default is './models/weights/court_kpRCNN.pth'.
  - kpRCNN_path: Path to Keypoint-RCNN's weight                       Default is './models/weights/kpRCNN.pth'.
  - opt_path: Path to transformer model's weight                      Default is './models/weights/OPT_16_head_dp.pt'.
  - scaler_path: Path to data scaler's file                           Default is './models/weights/scaler.pickle'.
- sa_queue length: Length of the sa_queue                             Default is 5.
- video_directory: Directory with unresolved videos                   Default is '../videos'.
- video_save_path: Directory to store resolved videos                 Default is '../outputs/videos'.
- joint_save_path: Directory to store player joints and frame info    Default is '../outputs/joints'.
- rally_save_path: Directory to store rally-wise info                 Default is '../outputs/rallies'.
```
## Run the Code
```
python main.py
```
## Options
```
--yaml_path        STR      Path to AI Coach setting yaml.     Default is "../configs/ai_coach.yaml".
```
