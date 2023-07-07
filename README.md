# Trandformer-and-RCNN-on-Badminton-Stroke-Direction-Identification
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
## Run the Code
```
python main.py
```
```
--yaml_path        STR      Path to AI Coach setting yaml.     Default is "../configs/ai_coach.yaml".
```
