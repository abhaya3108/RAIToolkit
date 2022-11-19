# Libraries used
import logging
from logging.handlers import TimedRotatingFileHandler

# Setting format for logger
FORMATTER = logging.Formatter("%(asctime)s | %(message)s", datefmt = '%Y-%m-%d|%H:%M:%S')
LOG_FILE = "../error_handler/log/error.log"


def get_file_handler():
    ''' This method configure the file handler '''
    
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='w0')
    file_handler.setFormatter(FORMATTER)
    return file_handler
    

def get_logger(logger_name):
    ''' Create handler and return logger '''
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_file_handler())
    logger.propogate = False
    return logger