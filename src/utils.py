import os

def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
            pass
    return paths

def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)
    return isExit