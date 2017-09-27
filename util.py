import os
import logging

fh = None
ch = None

def get_logger(app_name="Some Application", logfolder=".", fname="log_file.log"):
    """
    Returns a function which logs messages to console and also to a file (in PWD).

    For more info, see: https://docs.python.org/3.6/howto/logging-cookbook.html#logging-cookbook
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)
    
    global fh
    global ch

    if fh is not None:
        fh.flush()
        fh.close()
        print("Flushed and closed FH.")

    fh = logging.FileHandler(logfolder + os.sep + fname)
    fh.setLevel(logging.DEBUG)

    if ch is  None:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        print("Created new stream Handler.")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
