import re
import base64
import cStringIO as StringIO
from M2Crypto import RSA, EVP
import ConfigParser

# Read the license file
license_file_text = open("finmag-example.license").read()
m = re.match("(.*\n)license *= *(\\S+)\n", license_file_text, re.DOTALL)
if not m:
    raise Exception("Could not parse the license file")
license_text = m.group(1)
license_signature = base64.b64decode(m.group(2))

# Load the public key
# In the real code we might want to embed the public key into the compiled verification file
# so that the user could not change it
rsa = RSA.load_pub_key('example-public.pem')
digest = EVP.MessageDigest("sha256")
digest.update(license_text)
verified = rsa.verify_rsassa_pss(digest.digest(), license_signature, algo='sha256') != 0

print "License verified:", verified
# If verified, read the parameters
if verified:
    # We don't have to use ConfigParser here - can just parse the license using a regexp etc
    parser = ConfigParser.SafeConfigParser()
    parser.readfp(StringIO.StringIO("[dummy]\n" + license_text))
    expires = parser.get("dummy", "expires")
    licensed_to = parser.get("dummy", "licensed_to")
    print "Expires:", expires
    print "Licensed To:", licensed_to
