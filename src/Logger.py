import datetime

class Logger:
    log_level = 1

    def __init__(self, enabled, level = 1):
        self.enabled = enabled
        self.log_level = level
    
    def log(self, message, level = 1):
        if self.enabled == True and level >= self.log_level :
            print(f'{datetime.datetime.now().time()} - {message}')