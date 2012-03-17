import numpy


def energy_density():

    data = numpy.loadtxt('nmag/nmag-time-Eani-Eexch-step.txt')

    nm=1e-9
    crosssection = 1*nm*1*nm
    vol = crosssection*504*nm

    time = data[:,0]
    Eani = data[:,1]/vol
    Eexch = data[:,2]/vol
    step = data[:,3]

    return time, Eani, Eexch

if __name__=="__main__":
    print energy_density()



