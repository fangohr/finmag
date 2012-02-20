import socket

class Connection(object):
    def __init__(self, server):
        self._socket = socket.socket()
        self._socket.connect(server)
        self._socketfileobject = self._socket.makefile()

    def read(self):
        return self._socketfileobject.readline().rstrip()

    def send(self, *messages):
        for message in messages:
            self._socketfileobject.write(message + "\r\n")
        self._socketfileobject.flush()
