import importlib

def get_param_setting(set_name):
    params = importlib.import_module('setting.' + set_name)
    return params