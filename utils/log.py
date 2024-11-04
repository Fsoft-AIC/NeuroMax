import logging


def setup_logger(logger_name, log_file):
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def get_logger(logger_name):
    return logging.getLogger(logger_name)
