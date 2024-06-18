import logging
import os

def setupLogger(name):
    logsDir = 'logs'
    if not os.path.exists(logsDir):
        os.makedirs(logsDir)

    # Create a custom logger
    logger = logging.getLogger(name)
    
    # Set the overall logging level (this can be adjusted as needed)
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    consoleHandler = logging.StreamHandler()  # Handler for the console
    fileHandler = logging.FileHandler(os.path.join(logsDir, 'app.log'))  # Handler for the log file
    
    # Set levels for handlers
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler.setLevel(logging.DEBUG)
    
    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    
    return logger
