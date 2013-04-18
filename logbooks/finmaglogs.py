"""
Using CherryPy to put together very basic webserver to display
today's discussions on the IRC channel.

Hans Fangohr, 6 September 2012
"""

import os
import time
import cherrypy

REFRESHCODE = """<META HTTP-EQUIV="REFRESH" CONTENT="60"> """

class HelloWorld:
    def index(self):
        # CherryPy will call this method for the root URI ("/") and send
        # its return value to the client. Because this is tutorial
        # lesson number 01, we'll just send something really simple.
        # How about...
        return """Go to 
        <ul>
          <li> <a href="finmagirc">finmagirc</a> to see the 
               discussion today so far (updated every minute).<br>
          </li>
          <li><a href="finmagircupdate">finmagircupdate</a> to
               see the discussions today so far (updated when the page is loaded)
          </li>
        </ul>"""
    index.exposed=True

    def finmagirc(self):
        with open("/home/fangohr/www/ircdiff.txt","r") as f:
            text = f.read()
        return REFRESHCODE + text.replace('\n','<br>')
    finmagirc.exposed = True

    def finmagircupdate(self):
        os.system("/bin/sh /home/fangohr/bin/finmag-update-todays-irc-diff.sh > /home/fangohr/www/ircdiff.txt")
        return "Updating done ("+time.asctime()+")<br><br>"+\
            self.finmagirc()
        
    finmagircupdate.exposed = True

# CherryPy always starts with cherrypy.root when trying to map request URIs
# to objects, so we need to mount a request handler object here. A request
# to '/' will be mapped to cherrypy.root.index().
cherrypy.root = HelloWorld()

if __name__ == '__main__':
    cherrypy.config.update(file = 'cherrypy.conf')
    # Start the CherryPy server.
    cherrypy.server.start()

