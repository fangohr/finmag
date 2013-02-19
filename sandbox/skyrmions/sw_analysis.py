import numpy as np
import pylab as plt
import matplotlib.cm as cm

file_name = 'output.npz'
threshold = 1000
def find_max(a, threshold=0.5):
    index = []
    for i in range(1,len(a)-1):
        if a[i-1] < a[i] > a[i+1] and a[i]>threshold:
            index.append(i)
    return index

#load the file with tsim, mx, my and mz components
npzfile = np.load(file_name)

#extract numpy arrays from file
tsim = npzfile['tsim']
mx = npzfile['mx']
my = npzfile['my']
mz = npzfile['mz']

#caluclate and subtract the average value of magnetisation for each component
mx -= np.average(mx)
my -= np.average(my)
mz -= np.average(mz)

#Fourier transform complex numpy arrays for all sample point magnetisation components
fmx = np.zeros([mx.shape[0], mx.shape[1], mx.shape[2]], dtype=complex)
fmy = np.zeros([my.shape[0], my.shape[1], my.shape[2]], dtype=complex)
fmz = np.zeros([mz.shape[0], mz.shape[1], mz.shape[2]], dtype=complex)

#compute the FFT for all mesh points and populate arrays fmx, fmy and fmz
for i in range(mx.shape[0]):
    for j in range(mx.shape[1]):
        fmx[i,j,:] = np.fft.fft(mx[i,j,:])
        fmy[i,j,:] = np.fft.fft(my[i,j,:])
        fmz[i,j,:] = np.fft.fft(mz[i,j,:])
   
#compute the average of FFT     
#sum the fft arrays
s_fmx = 0
s_fmy = 0
s_fmz = 0
for i in xrange(mx.shape[0]):
    for j in xrange(mx.shape[1]):
        s_fmx += fmx[i,j,:]
        s_fmy += fmy[i,j,:]
        s_fmz += fmz[i,j,:]

#compute the average of FFT
fmx_av = s_fmx / mx.shape[0]*mx.shape[1]
fmy_av = s_fmy / mx.shape[0]*mx.shape[1]
fmz_av = s_fmz / mx.shape[0]*mx.shape[1]

#compute the frequency axis values
freq = np.fft.fftfreq(tsim.shape[-1], d=tsim[1]-tsim[0])

#create arrays for plotting (upper halves of arrays)
freq_plot = freq[0:len(freq)/2+1]
fmx_av_plot = np.absolute(fmx_av[0:len(fmx_av)/2+1])
fmy_av_plot = np.absolute(fmy_av[0:len(fmy_av)/2+1])
fmz_av_plot = np.absolute(fmz_av[0:len(fmz_av)/2+1])

#plot the average fft for all three components of magnetisation
#x component
plt.figure(1)
p1 = plt.subplot(311)
p1.plot(freq_plot, fmx_av_plot, label='mx')
plt.xlim([0,1e11])
plt.legend()
plt.xlabel('f')
plt.ylabel('|S|')
#y component
p2 = plt.subplot(312)
p2.plot(freq_plot, fmy_av_plot, label='my')
plt.xlim([0,1e11])
plt.legend()
plt.xlabel('f')
plt.ylabel('|S|')
#z component
p3 = plt.subplot(313)
p3.plot(freq_plot, fmz_av_plot, label='mz')
plt.xlim([0,1e11])
plt.legend()
plt.xlabel('f')
plt.ylabel('|S|')
#plt.show()

#find peaks in fft
mode_indices = find_max(fmz_av_plot, threshold = threshold)
print mode_indices

fig = 2
mode_number = 0
if mode_indices != []:
    for i in mode_indices:
        #plot the x component of the mode
        #title_string = 'frequency: ' + str(freq_plot[i]) + ', component: x'
        mode_x = np.absolute(fmx[:,:,i])
        mode_y = np.absolute(fmy[:,:,i])
        mode_z = np.absolute(fmz[:,:,i])

        plt.figure(2)
        plt.subplot(len(mode_indices),3,mode_number+1)
        plt.imshow(mode_x, cmap = cm.Greys_r)
        plt.subplot(len(mode_indices),3,mode_number+2)
        plt.imshow(mode_y, cmap = cm.Greys_r)
        plt.subplot(len(mode_indices),3,mode_number+3)
        plt.imshow(mode_z, cmap = cm.Greys_r)
        #plt.show()
        mode_number += 3

#show the graphs
plt.show()
