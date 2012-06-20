import os
import M2Crypto

M2Crypto.Rand.rand_seed (os.urandom(1024))
key_pair = M2Crypto.RSA.gen_key(1024, 65537, lambda x: None)
key_pair.save_key("example-private.pem", None) # Use empty password
key_pair.save_pub_key("example-public.pem")