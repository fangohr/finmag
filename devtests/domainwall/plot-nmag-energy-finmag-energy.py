import numpy


def energy_density_nmag():

    data = numpy.loadtxt('nmag/nmag-time-Eani-Eexch-step.txt')

    nm=1e-9
    crosssection = 1*nm*1*nm
    vol = crosssection*504*nm

    time = data[:,0]
    Eani = data[:,1]
    Eexch = data[:,2]
    step = data[:,3]
    print "final values Eani=%g" % Eani[-1]
    print "final values Eex=%g" % Eexch[-1]

    return time, Eani, Eexch

def energy_density_finmag():
    data = numpy.loadtxt('data.txt')
    time=data[:,0]
    nm=1e-9
    crosssection = 1
    
    Eani = data[:,1]*crosssection
    Eexch = data[:,2]*crosssection*-1
    return time,Eani,Eexch


if __name__=="__main__":
    t1,an1,ex1= energy_density_nmag()
    t2,an2,ex2= energy_density_finmag()
    import pylab
    #pylab.figure(1)
    pylab.plot(t1,an1,label='anisotropy nmag')
    pylab.plot(t1,ex1,label='exchange nmag')
    #pylab.figure(2)
    pylab.plot(t2,an2,label='anisotropy finmag')
    pylab.plot(t2,ex2,label='exchange finmag')
    pylab.axis([0,1e-10,-1e6,0.1e6])
    pylab.legend(loc='bottom')
    pylab.show()



