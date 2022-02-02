import os
from logging.config import dictConfig


__all__ = ['set_logging']


def set_logging():
    dictConfig(_CONFIG)


_CONFIG = {
    'version': 1,
    'formatters': {
        'console': {
            'format': '[%(asctime)s] %(levelname)s: %(message)s',
        },
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'stream': 'ext://sys.stdout',
        },
    },
    'root': {
        'handlers': ['stdout'],
        'level': os.environ.get('LOGLEVEL', 'INFO'),
    }
}
