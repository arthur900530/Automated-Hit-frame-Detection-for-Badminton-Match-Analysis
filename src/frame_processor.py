import torch

class FrameProcessor(object):
    '''
    Tasks involving Keypoint RCNNs
    '''
    def __init__(self, args):
        self.args = args
        self.drawn_img_list = []
        self.player_joint_list = []

    def setup_court_kpRCNN(self):
        self.court_kp_model = torch.load(self.args['court_kpRCNN_path'])
        self.court_kp_model.to(self.device).eval()