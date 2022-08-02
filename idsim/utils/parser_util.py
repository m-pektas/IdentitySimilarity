from munch import DefaultMunch

def dictToMunch(opt):
    return DefaultMunch.fromDict(opt)
