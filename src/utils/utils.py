import os


def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
    return paths

def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)
    return isExit

def search_target(target_paths, video_name):
    for t_path in target_paths:
        t_name = t_path.split('/')[-1]
        if video_name in t_name:
            return t_path
    return False




