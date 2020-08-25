# -*- coding: utf-8 -*-
import time
import socket
import logging
import subprocess

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

_logger = logging.getLogger(__name__)


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


class ProcessStopper(object):
    def __init__(self):
        self.debug = False
        self.ip = get_host_ip()

    def get_ids(self):
        output = subprocess.check_output(
            ["top", "-u", "mechmind", "-n", "1"], stderr=subprocess.STDOUT)
        output = output.decode()
        lines = output.split("\n")
        processing = []
        for i, line in enumerate(lines):
            line = line.replace('\x1b(B\x1b[m', '').replace('\x1b[1m', '').replace('\x1b[m\x0f', '')
            splits = line.split()[:-1]
            if i > 6 and len(splits) > 2:
                processing.append(splits)
        if self.debug:
            _logger.info(str(processing))
        kill_id = [line[0] for line in processing if line[-1] == "python"]
        return kill_id

    def kill(self):
        kill_id = self.get_ids()
        while kill_id:
            time.sleep(1)
            # kill
            for to_killed_id in kill_id:
                if self.debug:
                    _logger.info("Pid {}".format(to_killed_id))
                else:
                    _logger.info("Kill {}".format(to_killed_id))
                    subprocess.run(["kill", to_killed_id])
            kill_id = self.get_ids()


if __name__ == "__main__":
    stopper = ProcessStopper()
    stopper.kill()
