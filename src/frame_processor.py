import torch

class FrameProcessor(object):
    '''
    Tasks involving Keypoint RCNNs
    '''
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.drawn_img_list = []
        self.player_joint_list = []
        self.court_points = None
        self.setup_court_kpRCNN()

    def setup_court_kpRCNN(self):
        self.court_kp_model = torch.load(self.args['court_kpRCNN_path'])
        self.court_kp_model.to(self.device).eval()
    
    def get_court_data(self, img, frame_height):
        pass