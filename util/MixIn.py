from util.Logger import Logger


class LoggerMixIn:
    @property
    def log(self):
        return Logger(self.__class__.__name__)