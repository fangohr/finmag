import numpy as np
import pylab as plt

npzfile = np.load('output.npz')
mx = npzfile['mx']
my = npzfile['my']
mz = npzfile['mz']
tsim = npzfile['tsim']

mx -= np.average(mx)
my -= np.average(my)
mz -= np.average(mz)

spx = []
spy = []
spz = []

for i in range(mx.shape[0]):
    for j in range(mx.shape[1]):
        spx.append(np.fft.fft(mx[i,j,:]))
        spy.append(np.fft.fft(my[i,j,:]))
        spz.append(np.fft.fft(mz[i,j,:]))
        
sax = 0
say = 0
saz = 0
for i in xrange(mx.shape[0]*mx.shape[1]):
    sax += spx[i]
    say += spy[i]
    saz += spz[i]

sax /= mx.shape[0]*mx.shape[1]
say /= mx.shape[0]*mx.shape[1]
saz /= mx.shape[0]*mx.shape[1]

ftx = sax
fty = say
ftz = saz

freq = np.fft.fftfreq(tsim.shape[-1], d=9e-12)

f = freq[0:len(freq)/2+1]
fftx = np.absolute(ftx[0:len(ftx)/2+1])
ffty = np.absolute(fty[0:len(fty)/2+1])
fftz = np.absolute(ftz[0:len(ftz)/2+1])

plt.figure(1)
p1 = plt.subplot(311)
p1.plot(f, fftx, label='mx')
plt.xlim([0,1e10])
plt.legend()
plt.xlabel('f')
plt.ylabel('|S|^2')
p2 = plt.subplot(312)
p2.plot(f, ffty, label='my')
plt.xlim([0,1e10])
plt.legend()
plt.xlabel('f')
plt.ylabel('|S|^2')
p3 = plt.subplot(313)
p3.plot(f, fftz, label='mz')
plt.xlim([0,1e10])
plt.legend()
plt.xlabel('f')
plt.ylabel('|S|^2')
plt.show()

def find_max(a):
    index = []
    for i in range(1,len(a)-1):
        if a[i-1]<a[i]>a[i+1] and a[i]>2:
            index.append(i)
    return index

mode_indices = find_max(fftz)

print mode_indices

x_c = range(mx.shape[0])
y_c = range(mx.shape[1])

#for imode in mode_indices:
fig = 2
mode = np.zeros([mx.shape[0], mx.shape[1]])

for i in range(mx.shape[0]):
    for j in range(mx.shape[1]):
        ftrans = np.fft.fft(mz[i,j,:])
        mode[i,j] = np.absolute(ftrans[146])#**2

plt.figure(fig)

#X, Y = plt.meshgrid(x_c,y_c)
#plt.pcolor(X, Y, mode, cmap=plt.cm.RdBu) 
plt.imshow(mode)
plt.show()
fig += 1
