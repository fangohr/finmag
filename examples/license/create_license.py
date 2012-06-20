from M2Crypto import RSA, EVP
import base64
from datetime import datetime, timedelta

# Load the private key
rsa = RSA.load_key('example-private.pem')

# Set up the license text
EXPIRATION = str(datetime.now() + timedelta(days=1))
NAME = "University of Southampton"
license_text = "expires = %s\nlicensed_to = %s\n" % (EXPIRATION, NAME)

# Sign the license text
digest = EVP.MessageDigest("sha256")
digest.update(license_text)
signature = rsa.sign_rsassa_pss(digest.digest(), algo='sha256')

# Write the license and the signature
f = open("finmag-example.license", "w")
f.write(license_text)
f.write("license = %s\n" % base64.b64encode(signature))
f.close()