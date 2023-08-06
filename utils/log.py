# Some codes are borrowed from  https://github.com/XPixelGroup/BasicSR
import logging
import sys
from .dist import get_dist_info, master_only
from torch.utils.tensorboard import SummaryWriter

initialized_logger = {}

@master_only
def init_tb_logger(log_dir=None, comment=""):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir, comment=comment)
    return tb_logger

def get_logger(logger_name: str = "text", log_file: str = None, log_level=logging.INFO, rank=-1):

    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    formatstr = '[%(asctime)s] %(levelname)s: %(message)s'
    datefmt = '%Y-%m-%d %A %H:%M:%S'

    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(logging.Formatter(formatstr, datefmt=datefmt))
    # console_handler.setLevel(logging.INFO)
    # logging.basicConfig(level=logging.INFO, 
    #                     handlers=[file_handler, console_handler],
    #                     )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(formatstr, datefmt=datefmt))
    logger.addHandler(stream_handler)
    logger.propagate = False

    rank, _ = get_dist_info()
    if rank > 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(formatstr, datefmt=datefmt))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    # file_handler = logging.FileHandler(filename=log_file, mode='a')
    # file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S'))
    # file_handler.setLevel(logging.INFO)

    initialized_logger[logger_name] = True

    return logger

def get_tb_logger(log_dir=None, comment=""):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir, comment=comment)
    return tb_logger

def make_env_folder(opt: dict, mode: str = "train"):
    pass