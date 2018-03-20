import numpy as np

"""

h2 = h - np.dot(h,t)*t + sf*t

self.Heff[i][:] = h2[:]

where h, t are 1d numpy arrays, and sf is a double number, if I change the code to,


h2 = np.dot(h,t)*t

h3 = h - h2 + sf*t

self.Heff[i][:] = h3[:]

does anyone know why h2 = h - np.dot(h,t)*t + sf*t is not safe?  Weiwei

"""

def test_different_results():
    h = np.random.random(10000)
    t = np.random.random(10000)
    sf = np.random.random()

    one = h - np.dot(h, t) * t + sf * t
    
    temp = np.dot(h, t) * t
    two = h - temp + sf * t 

    print ""
    print one
    print two
    print "Maximum difference: ", np.abs(one - two).max()
    assert np.allclose(one, two, atol=1e-14, rtol=0)

if __name__ == "__main__":
    test_different_results()
