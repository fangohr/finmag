from bot import Bot

#SERVER = ("", 6667)
#IDENT = ("", "")
#CHANNEL = ("", "")
#
SERVER = ("irc.freenode.net", 6667)
IDENT = ("DokosBot", "D0k0s_B0t!")
CHANNEL = ("#finmag", "Soton2012!")

LOGGING_DIR = ""

if IDENT[0] != "":
    bot = Bot(SERVER, IDENT, CHANNEL, LOGGING_DIR)
    bot.loop()
else:
    print "Please configure."
