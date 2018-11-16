from models.utils import case_getattr
from importlib import import_module


def get_metrics(metrics):
    if type(metrics) == str:
        metrics = metrics.split(';')
    return [case_getattr(import_module('metrics.' + m), m) for m in metrics]
