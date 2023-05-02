from transformer import OptimusPrimeContainer
from sacnn import SACNNContainer
import utils
from PIL import Image
import cv2

class ShotAngleQueue(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.queue = []
    
    def push(self, frame_info):
        if len(self.queue) < self.max_len:
            self.queue.append(frame_info)
            return None
        else:
            first_info = self.queue.pop(0)
            self.queue.append(frame_info)
            return first_info
            # sa = first_info[0]
            # pil_image = first_info[1]
            # frame = first_info[2]
            # return sa, pil_image, frame

class VideoResolver(object):
    def __init__(self, args, video_dir):
        self.args = args
        self.get_videos(video_dir)
        self.setup_sa_queue()
        self.sacnn = SACNNContainer(self.args)
        # self.opt = OptimusPrimeContainer(self.args)

    def setup_sa_queue(self):
        self.sa_queue = ShotAngleQueue(self.args['saqueue length'])

    def get_videos(self, vid_dir):
        self.video_paths = utils.get_path(vid_dir)

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
                    frame_info = self.sa_queue.push((sa, pil_image, frame))
                    if frame_info:
                        sa, pil_image, frame = frame_info[0], frame_info[1], frame_info[2]
                    saved_count += 1
                    print(saved_count, ' / ', target_save_count)
                frame_count += 1
            else:
                break
        cap.release()

class AICoach(object):
    def __init__(self, args, video_dir):
        """
        """
        self.args = args
        self.setup_resolver(video_dir)
        
    def setup_resolver(self, video_dir):
        self.vid_resolver = VideoResolver(self.args, video_dir)
    
    def start_resolver(self):
        print('Start Resolving...')
        self.vid_resolver.start_resolve()
    

