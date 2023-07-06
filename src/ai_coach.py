from video_resolver import VideoResolver


class AICoach(object):
    def __init__(self, args):
        self.args = args
        self.__setup_resolver()
    
    def start_resolver(self):
        print('Start Resolving...')
        self.__vid_resolver.start_resolve()

    def __setup_resolver(self):
        self.__vid_resolver = VideoResolver(self.args)
    