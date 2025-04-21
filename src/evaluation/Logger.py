import time

class Logger(object):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, 'w')
        
    def log(self, msg):
        self.log_file.write(msg + '' ) 
        