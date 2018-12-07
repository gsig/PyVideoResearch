from models.utils import case_getattr
from importlib import import_module


def get_tasks(tasks):
    if tasks == '':
        return []
    if type(tasks) == str:
        tasks = tasks.split(';')
    return [case_getattr(import_module('tasks.' + m), m) for m in tasks]
