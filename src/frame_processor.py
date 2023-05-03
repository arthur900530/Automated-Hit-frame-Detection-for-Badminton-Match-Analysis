import torch
import torchvision
import numpy as np
import copy
from torchvision.transforms import functional as F


class FrameProcessor(object):
    '''
    Tasks involving Keypoint RCNNs
    '''
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.drawn_img_list = []
        self.player_joint_list = []
        self.got_info = False
        self.setup_court_kpRCNN()

    def setup_court_kpRCNN(self):
        self.court_kp_model = torch.load(self.args['court_kpRCNN_path'])
        self.court_kp_model.to(self.device).eval()
    
    def get_court_info(self, img, frame_height):
        print('get court info')
        img = F.to_tensor(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        output = self.court_kp_model(img)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        self.true_court_points = copy.deepcopy(keypoints[0])

        '''
        l -> left, r -> right, y = a * x + b
        '''
        l_a = (self.true_court_points[0][1] - self.true_court_points[4][1]) / (self.true_court_points[0][0] - self.true_court_points[4][0])
        l_b = self.true_court_points[0][1] - l_a * self.true_court_points[0][0]
        r_a = (self.true_court_points[1][1] - self.true_court_points[5][1]) / (self.true_court_points[1][0] - self.true_court_points[5][0])
        r_b = self.true_court_points[1][1] - r_a * self.true_court_points[1][0]
        mp_y = (self.true_court_points[2][1] + self.true_court_points[3][1]) / 2

        self.court_info = [l_a, l_b, r_a, r_b, mp_y]

        self.multi_points = self.partition(self.correction()).tolist()

        keypoints[0][0][0] -= 80
        keypoints[0][0][1] -= 80
        keypoints[0][1][0] += 80
        keypoints[0][1][1] -= 80
        keypoints[0][2][0] -= 80
        keypoints[0][3][0] += 80
        keypoints[0][4][0] -= 80
        keypoints[0][4][1] = min(keypoints[0][4][1] + 80, frame_height - 40)
        keypoints[0][5][0] += 80
        keypoints[0][5][1] = min(keypoints[0][5][1] + 80, frame_height - 40)

        self.extended_court_points = keypoints[0]

        self.got_info = True
        
    def correction(self):
        court_kp = np.array(self.true_court_points)
        ty = np.round((court_kp[0][1] + court_kp[1][1]) / 2)
        my = (court_kp[2][1] + court_kp[3][1]) / 2
        by = np.round((court_kp[4][1] + court_kp[5][1]) / 2)
        court_kp[0][1] = ty
        court_kp[1][1] = ty
        court_kp[2][1] = my
        court_kp[3][1] = my
        court_kp[4][1] = by
        court_kp[5][1] = by
        return court_kp

    def partition(self, court_kp):
        tlspace = np.array(
            [np.round((court_kp[0][0] - court_kp[2][0]) / 3), np.round((court_kp[2][1] - court_kp[0][1]) / 3)], dtype=int)
        trspace = np.array(
            [np.round((court_kp[3][0] - court_kp[1][0]) / 3), np.round((court_kp[3][1] - court_kp[1][1]) / 3)], dtype=int)
        blspace = np.array(
            [np.round((court_kp[2][0] - court_kp[4][0]) / 3), np.round((court_kp[4][1] - court_kp[2][1]) / 3)], dtype=int)
        brspace = np.array(
            [np.round((court_kp[5][0] - court_kp[3][0]) / 3), np.round((court_kp[5][1] - court_kp[3][1]) / 3)], dtype=int)

        p2 = np.array([court_kp[0][0] - tlspace[0], court_kp[0][1] + tlspace[1]])
        p3 = np.array([court_kp[1][0] + trspace[0], court_kp[1][1] + trspace[1]])
        p4 = np.array([p2[0] - tlspace[0], p2[1] + tlspace[1]])
        p5 = np.array([p3[0] + trspace[0], p3[1] + trspace[1]])

        p8 = np.array([court_kp[2][0] - blspace[0], court_kp[2][1] + blspace[1]])
        p9 = np.array([court_kp[3][0] + brspace[0], court_kp[3][1] + brspace[1]])
        p10 = np.array([p8[0] - blspace[0], p8[1] + blspace[1]])
        p11 = np.array([p9[0] + brspace[0], p9[1] + brspace[1]])

        kp = np.array([court_kp[0], court_kp[1],
                    p2, p3, p4, p5,
                    court_kp[2], court_kp[3],
                    p8, p9, p10, p11,
                    court_kp[4], court_kp[5]], dtype=int)

        ukp = []

        for i in range(0, 13, 2):
            sub2 = np.round((kp[i] + kp[i + 1]) / 2)
            sub1 = np.round((kp[i] + sub2) / 2)
            sub3 = np.round((kp[i + 1] + sub2) / 2)
            ukp.append(kp[i])
            ukp.append(sub1)
            ukp.append(sub2)
            ukp.append(sub3)
            ukp.append(kp[i + 1])
        ukp = np.array(ukp, dtype=int)
        return ukp