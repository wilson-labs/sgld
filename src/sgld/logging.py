import logging
from logging.config import dictConfig
import os
from pathlib import Path
import time


__all__ = ['set_logging']


class MetricsFilter(logging.Filter):
    def __init__(self, extra_key='metrics', invert=False):
        super().__init__()
        self.extra_key = extra_key
        self.invert = invert

    def filter(self, record):
        should_pass = hasattr(record, self.extra_key) and getattr(record, self.extra_key)
        if self.invert:
            should_pass = not should_pass
        return should_pass


class MetricsFileHandler(logging.FileHandler):
    def emit(self, record):
        if hasattr(record, 'prefix'):
            record.msg = {f'{record.prefix}/{k}': v for k, v in record.msg.items()}
        record.msg['timestamp_ns'] = time.time_ns()
        return super().emit(record)


def set_logging(root, metrics_extra_key='metrics'):
    _CONFIG = {
        'version': 1,
        'formatters': {
            'console': {
                'format': '[%(asctime)s] (%(funcName)s:%(levelname)s) %(message)s',
            },
        },
        'filters': {
            'metrics': {
                '()': MetricsFilter,
                'extra_key': metrics_extra_key,
            },
            'nometrics': {
                '()': MetricsFilter,
                'extra_key': metrics_extra_key,
                'invert': True,
            },
        },
        'handlers': {
            'stdout': {
                '()': logging.StreamHandler,
                'formatter': 'console',
                'stream': 'ext://sys.stdout',
                'filters': ['nometrics'],
            },
            'metrics_file': {
                '()': MetricsFileHandler,
                'filename': str(Path(root) / 'metrics.log'),
                'filters': ['metrics'],
            },
        },
        'loggers': {
            '': {
                'handlers': ['stdout', 'metrics_file'],
                'level': os.environ.get('LOGLEVEL', 'INFO'),
            },
        },
    }

    dictConfig(_CONFIG)