"""This package options includes option modules: task options, training options, test options, and basic options (used in both training and test)."""

import importlib

def find_task_using_name(task_name):
    """Import the module "options/[task_name]_options.py"."""

    task_filename = "options." + task_name + "_options"
    tasklib = importlib.import_module(task_filename)
    target_task_name = task_name.replace('_', '') + 'options'
    task = None
    for name, cls in tasklib.__dict__.items():
        if name.lower() == target_task_name.lower():
            task = cls

    if task is None:
        print("In %s.py, there should be a class with class name that matches %s in lowercase." % (task_filename, target_task_name))
        exit(0)

    return task

def get_option_setter(task_name):
    """Return the static method <modify_commandline_options> of the model class."""
    task_class = find_task_using_name(task_name)
    return task_class.modify_commandline_options
