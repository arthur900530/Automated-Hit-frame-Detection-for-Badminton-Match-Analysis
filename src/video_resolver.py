from transformer import OptimusPrimeContainer
from sacnn import SACNNContainer
from rally_processor import RallyProcessor
import utils
from PIL import Image
import cv2
import json

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
            sa, sa_condition = self.__check_sa_conditon(first_info[0])
            self.last_sa = sa
            first_info[0] = sa

            self.queue.append(frame_info)

            return first_info, sa_condition

    def get(self, index):
        return self.queue[index]

    def __check_sa_conditon(self, sa):
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
        self.__setup_videos()
        self.__setup_sa_queue()
        self.__setup_rally_processor()
        self.__sacnn = SACNNContainer(self.args)
        self.__opt = OptimusPrimeContainer(self.args)

    def start_resolve(self):
        for path in self.__video_paths:
            self.__resolve(path)

    def __setup_sa_queue(self):
        self.__sa_queue = ShotAngleQueue(self.args['saqueue length'])
    
    def __setup_rally_processor(self):
        self.__rally_processor = RallyProcessor(self.args)

    def __setup_videos(self):
        self.__video_paths = utils.get_path(self.args['video_directory'])

    def __resolve(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        total_frame_count = int(cap.get(7))
        fps = cap.get(5)

        time_rate = 0.1
        frame_rate = round(int(fps) * time_rate)
        frame_count, saved_count = 0, 0
        target_save_count = int(total_frame_count / frame_rate)

        zero_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_count % frame_rate == 0:
                    seceneImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sa = self.__sacnn.predict(seceneImg)
                    frame_info, sa_condition = self.__sa_queue.push([sa, frame])
                    '''
                    sa_condition:
                    0  ->  0 to 0
                    1  ->  0 to 1
                    2  ->  1 to 1
                    3  ->  1 to 0
                    '''
                    stat_code = 'Added to Queue'
                    if frame_info:
                        sa, frame = frame_info[0], frame_info[1]
                        if sa_condition == 0:
                            stat_code = '0 -> 0'
                            zero_count += 1                            
                        if sa_condition == 1:
                            stat_code = '0 -> 1'
                            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                            if not self.__rally_processor.got_info:
                                self.__rally_processor.get_court_info(self.__sa_queue.get(2)[-1], frame_height)
                            self.__rally_processor.add_frame(frame, frame_num)
                            rally_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                        if sa_condition == 2:
                            stat_code = '1 -> 1'
                            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                            self.__rally_processor.add_frame(frame, frame_num)

                        if sa_condition == 3:
                            stat_code = '1 -> 0'
                            rally_end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                            drawn_img_list, rally_info = self.__rally_processor.start_new_rally(rally_start_frame, rally_end_frame)
                            joint_sequence = rally_info['joints']
                            shuttle_flying_seq = self.__opt.predict(joint_sequence)
                            print(shuttle_flying_seq)
                            
                            with open(f"{self.args['joints_save_path']}/rally_{rally_info['rally count']}.json", 'w', encoding="utf-8") as f:
                                json.dump(rally_info, f, indent=2, ensure_ascii=False)

                            out = cv2.VideoWriter(f"{self.args['video_save_path']}/video_{rally_info['rally count']}.mp4",
                                  cv2.VideoWriter_fourcc(*'mp4v'), int(fps / frame_rate), (frame_width, frame_height))
                            
                            self.__create_video(out, drawn_img_list)
                            out.release()

                    saved_count += 1
                    print(f'{saved_count} / {target_save_count}, {stat_code}')
                frame_count += 1
            else:
                break
        cap.release()

    def __create_video(self, out, drawn_img_list):
        for i in range(len(drawn_img_list)):
            out.write(drawn_img_list[i])