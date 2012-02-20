import re
from connection import Connection
from logging import Logging

class Bot(object):
    ping_pattern = re.compile('^PING (?P<payload>.*)')
    chanmsg_pattern = re.compile(':(?P<nick>.*?)!\S+\s+?PRIVMSG\s+#(?P<channel>[-\w]+)\s+:(?P<message>[^\n\r]+)')

    def __init__(self, server, ident, channel, directory):
        self._dispatch_table = (
            (self.ping_pattern, self.handle_ping),
            (self.chanmsg_pattern, self.handle_chanmsg))
    
        self._logger = Logging(directory)
        self._connection = Connection(server)
        self.register_connection(ident)
        self.join_channel(channel)

    def loop(self):
        while True:
            line = self._connection.read()
            for pattern, handler in self._dispatch_table:
                match = pattern.match(line)
                if match:
                    handler(**match.groupdict())

    def handle_ping(self, payload):
        self._connection.send("PONG " + payload)
    
    def handle_chanmsg(self, nick, channel, message):
        self._logger.write(nick + ": " + message)

    def register_connection(self, ident):
        nick, passw = ident
        self._connection.send("PASS " + passw)
        self._connection.send("NICK " + nick)
        self._connection.send("USER " + nick + " 0 * :" + nick)

    def join_channel(self, channel):
        chan, passw = channel
        self._connection.send("JOIN " + chan + " " + passw)
