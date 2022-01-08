from . import *


def get_model(model_name):
    print('==> Geting model..')
    return eval(model_name)
