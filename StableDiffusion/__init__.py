import os
def GetStableDiffusionRootPath():
    return os.path.abspath(os.path.dirname(__file__));

def GetStableDiffusionSubpath(path):
    return os.path.join(GetStableDiffusionRootPath(), path);