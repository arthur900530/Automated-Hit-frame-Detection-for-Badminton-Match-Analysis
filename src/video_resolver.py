from transformer import OptimusPrimeContainer
from sacnn import SACNNContainer
from frame_processor import FrameProcessor
import utils
from PIL import Image
import cv2


class ShotAngleQueue(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.queue = []
        self.last_sa = 0
    
    def push(self, frame_info):
        sa_condition = None
        if len(self.queue) < self.max_len:        # queue is not full
            self.queue.append(frame_info)
            return None, None
        else:
            first_info = self.queue.pop(0)
            
            # if first_info[0] != self.last_sa:
            sa, sa_condition = self.check_sa_conditon(first_info[0])
            self.last_sa = sa
            first_info[0] = sa

            self.queue.append(frame_info)

            return first_info, sa_condition

    def check_sa_conditon(self, sa):
        '''
        return sa, cond in {0, 1, 2, 3}
        cond :  last sa  ->   sa
          0  :      0    ->   0
          1  :      0    ->   1
          2  :      1    ->   1
          3  :      1    ->   0
        '''
        sum = 0 + sa
        if self.last_sa == 1 and sa == 0:
            for info in self.queue:
                sum += info[0]
            if sum <= (self.max_len / 2):
                return 0, 3
            else:
                return 1, 2
        elif self.last_sa == 0 and sa == 1:
            for info in self.queue:
                sum += info[0]
            if sum >= (self.max_len / 2):
                return 1, 1
            else:
                return 0, 0
        elif self.last_sa == 1 and sa == 1:
            return 1, 2
        elif self.last_sa == 0 and sa == 0:
            return 0, 0
        

class VideoResolver(object):
    def __init__(self, args):
        self.args = args
        self.get_videos()
        self.setup_sa_queue()
        self.sacnn = SACNNContainer(self.args)
        # self.opt = OptimusPrimeContainer(self.args)

    def setup_sa_queue(self):
        self.sa_queue = ShotAngleQueue(self.args['saqueue length'])

    def get_videos(self):
        self.video_paths = utils.get_path(self.args['video_directory'])

    def start_resolve(self):
        for path in self.video_paths:
            self.resolve(path)

    def resolve(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        total_frame_count = int(cap.get(7))
        fps = cap.get(5)
        time_rate = 0.1
        frame_rate = round(int(fps) * time_rate)
        frame_count, saved_count = 0, 0
        target_save_count = int(total_frame_count / frame_rate)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_count % frame_rate == 0:
                    seceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sa = self.sacnn.predict(seceneImg)
                    frame_info, sa_condition = self.sa_queue.push([sa, pil_image, frame])
                    '''
                    sa_condition:
                    0  ->  0 to 0
                    1  ->  0 to 1
                    2  ->  1 to 1
                    3  ->  1 to 0
                    '''
                    stat_string = 'Added to Queue'
                    if frame_info:
                        sa, pil_image, frame = frame_info[0], frame_info[1], frame_info[2]
                        if sa_condition == 0:
                            stat_string = '0 -> 0'
                        if sa_condition == 1:
                            stat_string = '0 -> 1'
                        if sa_condition == 2:
                            stat_string = '1 -> 1'
                        if sa_condition == 3:
                            stat_string = '1 -> 0'

                    saved_count += 1
                    print(f'{saved_count} / {target_save_count}, {stat_string}')
                frame_count += 1
            else:
                break
        cap.release()
