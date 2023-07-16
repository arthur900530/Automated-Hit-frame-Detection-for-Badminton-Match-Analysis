import torch
import torchvision
import numpy as np
import copy
import cv2
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from models.transformer import OptimusPrimeContainer



class RallyProcessor(object):
    '''
    Tasks involving Keypoint RCNNs
    '''
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.got_info = False
        self.rally_count = 0
        self.__setup_sotrage_lists()
        self.__setup_RCNN()
        self.__setup_opt()
    
    def reset(self):
        self.got_info = False
        self.rally_count = 0
        self.__setup_sotrage_lists()
        self.rally_info = None

    def __setup_RCNN(self):
        self.__court_kpRCNN = torch.load(self.args['court_kpRCNN_path'])
        self.__court_kpRCNN.to(self.device).eval()
        self.__kpRCNN = torch.load(self.args['kpRCNN_path'])
        self.__kpRCNN.to(self.device).eval()
    
    def __setup_opt(self):
        self.__opt = OptimusPrimeContainer(self.args)

    def __setup_sotrage_lists(self):
        self.drawn_img_list = []
        self.player_joint_list = []
        self.frame_num_list = []
        self.start_end_frame_list = []

    def get_court_info(self, img, frame_height):
        img = F.to_tensor(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        output = self.__court_kpRCNN(img)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        self.__true_court_points = copy.deepcopy(keypoints[0])

        '''
        l -> left, r -> right, y = a * x + b
        '''
        l_a = (self.__true_court_points[0][1] - self.__true_court_points[4][1]) / (self.__true_court_points[0][0] - self.__true_court_points[4][0])
        l_b = self.__true_court_points[0][1] - l_a * self.__true_court_points[0][0]
        r_a = (self.__true_court_points[1][1] - self.__true_court_points[5][1]) / (self.__true_court_points[1][0] - self.__true_court_points[5][0])
        r_b = self.__true_court_points[1][1] - r_a * self.__true_court_points[1][0]
        mp_y = (self.__true_court_points[2][1] + self.__true_court_points[3][1]) / 2

        self.__court_info = [l_a, l_b, r_a, r_b, mp_y]

        self.__multi_points = self.__partition(self.__correction()).tolist()

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

        self.__extended_court_points = keypoints[0]

        self.got_info = True

    def add_frame(self, frame, frame_num):
        outputs = self.__human_detection(frame)
        result = self.__player_detection(outputs)
        if result:
            position, filtered_outputs = result
            for points in filtered_outputs:
                for i, joints in enumerate(points):
                    points[i] = joints[0:2]
            self.player_joint_list.append(filtered_outputs)
            self.frame_num_list.append(frame_num)
            frame = self.__draw_key_points(position, filtered_outputs, frame)

        self.drawn_img_list.append(frame)

    def start_new_rally(self, rally_start_frame, rally_end_frame):
        self.rally_count += 1
        drawn_img_list = copy.deepcopy(self.drawn_img_list)
        player_joint_list = copy.deepcopy(self.player_joint_list)
        frame_num_list = copy.deepcopy(self.frame_num_list)
        self.drawn_img_list = []
        self.player_joint_list = []
        self.frame_num_list = []
        
        result = self.__predict_flying_direction(player_joint_list, frame_num_list)

        if result:
            '''
            determine stroke & movement
            '''
            self.start_end_frame_list.append((rally_start_frame, rally_end_frame, self.rally_count))
            shuttle_flying_seq, hit_frames = result
            self.rally_info = {
                    'rally count': self.rally_count,
                    'start frame': rally_start_frame,
                    'end frame': rally_end_frame,
                    'shuttle directions': shuttle_flying_seq,
                    'hit frames': hit_frames,
                    'joints': player_joint_list
            }
            return (drawn_img_list, self.rally_info)
        else:
            return False
    
    def __check_valid_rally(self, joint_sequence):
        return False if len(joint_sequence) < 10 else True
    
    def __check_valid_sequence(self, shuttle_flying_seq):
        zero_count = 0
        for direction in shuttle_flying_seq:
            if direction == 0:
                zero_count += 1
        return False if zero_count/len(shuttle_flying_seq) > 0.6 else True

    def __predict_flying_direction(self, joint_sequence, frame_num_list):
        if self.__check_valid_rally(joint_sequence):
            shuttle_flying_seq = list(self.__opt.predict(joint_sequence).cpu().numpy().astype(float))
        else:
            return False
        
        if self.__check_valid_sequence(shuttle_flying_seq):
            hit_frame_indices = list(self.__check_hit_frame(shuttle_flying_seq))
            hit_frames = [frame_num_list[i] for i in hit_frame_indices]
        else:
            return False
        
        return (shuttle_flying_seq, hit_frames)
    
    def __check_hit_frame(self, direction_list):
        '''
        0 -> 1
        0 -> 2
        1 -> 2
        2 -> 1
        '''
        last_direction = 0
        hit_frame_indices = []
        for i in range(len(direction_list)):
            direction = direction_list[i]
            if direction != last_direction:
                if last_direction == 0:
                    hit_frame_indices.append(i)
                elif last_direction == 1:
                    if direction == 2:
                        hit_frame_indices.append(i)
                    else:
                        continue
                elif last_direction == 2:
                    if direction == 1:
                        hit_frame_indices.append(i)
                    else:
                        continue
            last_direction = direction
        hit_frame_indices = hit_frame_indices

        return hit_frame_indices

    def __draw_key_points(self, position, filtered_outputs, image):
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (11, 12), (5, 7),
                 (7, 9), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16), (5, 6)]
        c_edges = [[0, 1], [0, 5], [1, 2], [1, 6], [2, 3], [2, 7], [3, 4], [3, 8], [4, 9],
                   [5, 6], [5, 10], [6, 7], [6, 11], [7, 8], [7, 12], [8, 9], [8, 13], [9, 14],
                   [10, 11], [10, 15], [11, 12], [11, 16], [12, 13], [12, 17], [13, 14], [13, 18],
                   [14, 19], [15, 16], [15, 20], [16, 17], [16, 21], [17, 18], [17, 22], [18, 19],
                   [18, 23], [19, 24], [20, 21], [20, 25], [21, 22], [21, 26], [22, 23], [22, 27],
                   [23, 24], [23, 28], [24, 29], [25, 26], [25, 30], [26, 27], [26, 31], [27, 28],
                   [27, 32], [28, 29], [28, 33], [29, 34], [30, 31], [31, 32], [32, 33], [33, 34]]
        top_color_edge = (255, 0, 0)
        bot_color_edge = (0, 0, 255)
        top_color_joint = (115, 47, 14)
        bot_color_joint = (35, 47, 204)
        court_color_edge = (53, 195, 242)
        court_color_kps = (5, 135, 242)

        for i in range(2):
            pos = position[i]
            color = top_color_edge if i == 0 else bot_color_edge
            color_joint = top_color_joint if i == 0 else bot_color_joint
           
            keypoints = np.array(filtered_outputs[pos])  # 17, 2
            keypoints = keypoints[:, :].reshape(-1, 2)
            overlay = image.copy()

            # draw the court
            for e in c_edges:
                cv2.line(overlay, (int(self.__multi_points[e[0]][0]), int(self.__multi_points[e[0]][1])),
                                  (int(self.__multi_points[e[1]][0]), int(self.__multi_points[e[1]][1])),
                                  court_color_edge, 2, lineType=cv2.LINE_AA)
            for kps in [self.__multi_points]:
                for kp in kps:
                    cv2.circle(overlay, tuple(kp), 2, court_color_kps, 10)
                    
            alpha = 0.4
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            for p in range(keypoints.shape[0]):
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, color_joint, thickness=-1,
                            lineType=cv2.FILLED)

            for e in edges:
                cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                                (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                                color, 2, lineType=cv2.LINE_AA)
        return image
  
    def __human_detection(self, frame):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        t_image = transforms.Compose([transforms.ToTensor()])(pil_image).unsqueeze(0).to(self.device)
        outputs = self.__kpRCNN(t_image)
        return outputs

    def __player_detection(self, outputs):
        boxes = outputs[0]['boxes'].cpu().detach().numpy()
        filtered_joint = []
        joints = outputs[0]['keypoints'].cpu().detach().numpy()      
        in_court_indices = self.__check_in_court_instances(joints)

        if in_court_indices:
            conform, combination = self.__check_top_bot_court(in_court_indices, boxes)
            if conform:
                filtered_joint.append(joints[in_court_indices[combination[0]]].tolist())
                filtered_joint.append(joints[in_court_indices[combination[1]]].tolist())
                position = self.__top_bottom(filtered_joint)
                return (position, filtered_joint)
            else:
                return None
        else:
            return None

    def __check_top_bot_court(self, indices, boxes):
        '''
        check if up court and bot court got player
        '''
        court_mp = self.__court_info[4]
        for i in range(len(indices)):
            combination = 1
            if boxes[indices[0]][1] < court_mp < boxes[indices[combination]][3]:
                return True, [0, combination]
            elif boxes[indices[0]][3] > court_mp > boxes[indices[combination]][1]:
                return True, [0, combination]
            else:
                combination += 1
        return False, [0, 0]

    def __check_in_court_instances(self, joints):
        indices = []
        for i in range(len(joints)):
            if self.__in_court(joints[i]):
                indices.append(i)
        return None if len(indices) < 2 else indices
    
    def __in_court(self, joint):
        '''
        check if player is in court
        '''
        l_a = self.__court_info[0]
        l_b = self.__court_info[1]
        r_a = self.__court_info[2]
        r_b = self.__court_info[3]

        ankle_x = (joint[15][0] + joint[16][0]) / 2
        ankle_y = (joint[15][1] + joint[16][1]) / 2

        top = ankle_y > self.__extended_court_points[0][1]
        bottom = ankle_y < self.__extended_court_points[5][1]

        lmp_x = (ankle_y - l_b) / l_a
        rmp_x = (ankle_y - r_b) / r_a
        left = ankle_x > lmp_x
        right = ankle_x < rmp_x

        if left and right and top and bottom:
            return True
        else:
            return False

    def __top_bottom(self, joint):
        a = joint[0][-1][1] + joint[0][-2][1]
        b = joint[1][-1][1] + joint[1][-2][1]
        if a > b:
            top = 1
            bottom = 0
        else:
            top = 0
            bottom = 1
        return top, bottom
        
    def __correction(self):
        court_kp = np.array(self.__true_court_points)
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

    def __partition(self, court_kp):
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