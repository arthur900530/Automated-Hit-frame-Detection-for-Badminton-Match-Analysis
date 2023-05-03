from video_resolver import VideoResolver


class AICoach(object):
    def __init__(self, args):
        """
        """
        self.args = args
        self.setup_resolver()
        
    def setup_resolver(self):
        self.vid_resolver = VideoResolver(self.args)
    
    def start_resolver(self):
        print('Start Resolving...')
        self.vid_resolver.start_resolve()
    

