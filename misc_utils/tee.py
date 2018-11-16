"""
Implements a crude stdout-to-file redirect for keep history of experiments
The following code initializes the redirect:
import tee
tee.Tee(filename)
"""
import logging
import sys


class StreamToLogger(object):
    def __init__(self, stream, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        self.stream = stream

    def write(self, buf):
        self.stream.write(buf)
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        self.stream.flush()


class Tee(object):
    def __init__(self, filename):
        self.filename = filename
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(message)s',
            filename=filename,
            filemode='a'
        )
        stdout_logger = logging.getLogger('STDOUT')
        sl = StreamToLogger(sys.stdout, stdout_logger, logging.INFO)
        sys.stdout = sl

        stderr_logger = logging.getLogger('STDERR')
        sl = StreamToLogger(sys.stderr, stderr_logger, logging.ERROR)
        sys.stderr = sl
        print("Logging to file {}".format(filename))
