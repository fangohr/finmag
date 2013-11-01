import dolfin as df
import finmag

import numpy as np
import pylab
import scipy.optimize

"""

This file contains three functions that use finmag simulations on a 1D mesh
to obtain certain material characteristics. It also contains an example
showing usage of two of the functions. These function "declarations" are:

Find_Helix_Length( d, a, ms, h, tol=1e-9 )
Find_DMI( a, ms, l, h, d0=None, tol=1e-6 )

"""


def Find_Helix_Length(D, A, Ms, H):
    """Function that takes some material parameters and returns the
    estimated helix length.

    @parameters
    A   : Isotropic exchange energy constant (J/m)
    D   : Dzyaloshinskii-Moriya exchange energy constant (J/m^2)
    Ms  : Saturation magnetisation (A/m)
    H   : External magnetic field strength (a three-dimensional array) (A/m)
    """

    #Make a mesh, with lengths measured in unitLength.
    unitLength = 1e-9
    meshX = 1000  # Length of mesh (nm). Would prefer this to be an even
                  # number for the preceeding calculation.
    meshXHalf = meshX / 2
    meshN = 1000  # Number of points in the desired mesh.
    mesh = df.IntervalMesh(meshN - 1, -meshXHalf, meshXHalf)

    #Creating simulation object.
    simName = "Finding_Helix_Length"
    sim = finmag.Simulation(mesh, Ms, name=simName, unit_length=unitLength)

    #Create energy objects and add them to the simulation.
    eA = finmag.energies.Exchange( A ) #Isotropic exchange interaction energy object to use in the simulation.
    eD = finmag.energies.DMI( D )      #Dzyaloshinskii-Moriya exchange interaction energy object to use in the simulation.
    eH = finmag.energies.Zeeman( H )   #Zeeman energy object to use in the simulation.
    sim.add( eA ); sim.add( eD ); sim.add( eH )

    #Define initial magnetisation and set it.
    np.random.seed( 1 )
    def m_rand( pos ):
        """This function returns a consistent random vector direction."""
        out = np.random.random( 3 ) - 0.5
        return out / np.linalg.norm( out )

    sim.set_m( m_rand )

    #Run the simulation.
    tEnd = 5e-9 #Time the simulation will take (s)
    sim.run_until( tEnd )

    #Extract the magnetisation vectors of the relaxed mesh.
    xs = np.ndarray([ meshN ])
    ys = np.ndarray([ meshN ])
    zs = np.ndarray([ meshN ])

    for zI in xrange( meshN ):
        xs[ zI ] = sim.m[ zI           ]
        ys[ zI ] = sim.m[ zI +   meshN ]
        zs[ zI ] = sim.m[ zI + 2*meshN ]

    #Check to see if the ferromagnetic state has been encountered. This corresponds to all vectors having a strong component perpendicular to the helix in the direction of the external field. If the ferromagnetic state has been encountered, an exception should be raised.
    ferromagnetic = True
    for zI in xs:
        if abs( zI ) < 0.3:
            ferromagnetic = False
            break
    if ferromagnetic == False:
        ferromagnetic = True
        for zI in ys:
            if abs( zI ) < 0.3:
                ferromagnetic = False
                break
        if ferromagnetic == False:
            ferromagnetic = True
            for zI in zs:
                if abs( zI ) < 0.3:
                    ferromagnetic = False
                    break

    if ferromagnetic == True:
        msg = "Ferromagnetic relaxed state encountered. This suggests"+\
            "that the external magnetic field is too strong for these "+\
            "material parameters (D = {:.2e} J/m^2, A = {:.2e} J/m, Ms"+\
            " = {:.2e} A/m, h = ({:.2e}, {:.2e}, {:.2e}) A/m.".format(
            D, A, Ms, H[0], H[1], H[2] )
        raise ValueError(msg)
    #Find the fourier transform of the two magnetisation vector components.
    finmag.logger.info( "Calculating the fourier transform of magnetisation data." )
    ffty = np.fft.fft( ys )
    ffty = abs( ffty[ :len( ffty ) / 2 ] )

    fftz = np.fft.fft( zs )
    fftz = abs( fftz[ :len( fftz ) / 2 ] )

    #Calculate the discrete wavenumber domain fs of the magnetisation data after it is transformed.
    fPrecision = 1 / ( meshX * unitLength )
    fs = np.linspace( 0, meshN, meshN ) * fPrecision
    fs = fs[ :len( fs ) / 2 ]

    #Find the wavenumber peak that corresponds to the helix length.
    ly = fs[ list( ffty ).index( max( ffty ) ) ]
    lz = fs[ list( fftz ).index( max( fftz ) ) ]

    # #Do some plotting.
    # meshXs = np.linspace( -meshXHalf, meshXHalf, meshN )

    # pylab.figure( 1 )
    # pylab.plot( meshXs * unitLength, ys, 'x-' )
    # pylab.plot( meshXs * unitLength, zs, 'x-' )
    # pylab.xlabel( "Distance from centre of mesh (m)" )
    # pylab.ylabel( "Magnetisation component" )
    # pylab.axis( "tight" )
    # pylab.savefig( "mt.png" )

    # pylab.figure( 2 )
    # pylab.semilogy( fs, ffty, 'x-' )
    # pylab.xlabel( "So-called frequency (m-1)")
    # pylab.ylabel( "Discrete fourier transform of the magnetisation data." )
    # pylab.axis( "tight" )
    # pylab.savefig( "mf.png" )

    #Calculate the wavenumber that matches the waveform, as well as the helix length (analogous to the period).
    fOut = (ly+lz)/2.
    out = 1/fOut

    #Cleanup and return
    #print( "D = {:.4e} J/m^2. Helix length = {:.4e} m.".format( d, out ) )
    return out

def Find_DMI( A, Ms, l, H, D0=None, tol=1e-6, verbose=False ):
    """Function that takes some material parameters and returns the estimated DMI constant correct to a given tolerance.

    @parameters
    A       : Isotropic exchange energy constant (J/m)
    Ms      : Saturation magnetisation (A/m)
    l       : Helix length (m)
    H       : External magnetic field strength (a three-dimensional array) (A/m)
    D0      : Estimated Dzyaloshinskii-Moriya exchange energy constant (J/m^2), if any.
    tol     : Tolerance to which the DMI constant should be found, if any.
    verbose : Boolean to dictate whether or not simulation output is provided.
    """

    if( verbose == False ):
        logLevel = finmag.logger.level
        finmag.logger.setLevel( finmag.logging.ERROR )

    def Find_DMI_Signchange( A, Ms, l, H, D0, tol ):
        """Function that takes some material parameters and returns a range in which the DMI value can exist.

        @parameters
        A   : Isotropic exchange energy constant (J/m)
        Ms  : Saturation magnetisation (A/m)
        l   : Helix length (m)
        H   : External magnetic field strength (a three-dimensional array) (A/m)
        H0  : Estimated Dzyaloshinskii-Moriya exchange energy constant (J/m^2).
        tol : Tolerance to which the DMI constant should be found.
        """

        #Two values of d (with a helix lengths greater than and less than the desired helix length l) need to be found.
        #Initialise two arrays to hold d and l values,
        ds = []
        ls = []
        ds.append( d0 )

        #Find the sign of the evaluated helix length for the d0 guess, subtracted from the actual helix length.
        ls.append( Find_Helix_Length( ds[0], a, ms, h ) - l )

        #Find an increment size bigger than the desired tolerance to search for these lengths. The increment should be positive if the calculated length is too small and vice versa (due to the nature of the relationship).
        if ls[0] > 0:
            dIncrement = tol * -1
        else:
            dIncrement = tol

        #Find the sign of the evaluated helix length for another guess that's a bit far out from the desired tolerance, subtracted from the actual helix length.
        ds.append( d0 + dIncrement )
        ls.append( Find_Helix_Length( ds[1], a, ms, h ) - l )

        #Keep doing this until two different values have been found.
        while ls[-1] == ls[-2]:
            ds.append( ds[-1] + dIncrement )
            ls.append( Find_Helix_Length( ds[-1], a, ms, h ) - l )

        #Once the second value has been found, see if the sign change is different. If it is, then use those two as the interval. If not, proceed searching in the other direction until this situation is encountered.
        dRange = [0,0]
        if ls[-1] * ls[-2] < 0:
            dRange[0] = ds[-2]
            dRange[1] = ds[-1]
        else:
            #It's unfortunate, but now we must do the same as before, but in reverse.
            dIncrement *= -1
            ds = [ds[0]]
            ls = [ls[0]]

            #Find the sign of the evaluated helix length for another guess that's a bit far out from the desired tolerance, subtracted from the actual helix length.
            ds.append( d0 + dIncrement )
            ls.append( Find_Helix_Length( ds[1], a, ms, h ) - l )

            #Keep doing this until two different values have been found.
            while ls[-1] == ls[-2]:
                ds.append( ds[-1] + dIncrement )
                ls.append( Find_Helix_Length( ds[-1], a, ms, h ) - l )

            #Pray that a sign change has been found this time.
            if ls[-1] * ls[-2] >= 0:
                raise RuntimeError( "D Range cannot be found for the provided value of l!" )
            else:
                dRange[0] = ds[-2]
                dRange[1] = ds[-1]

        #Now that the range has been found, it can be returned.
        return dRange

    #===Find DMI function starts here...===#

    #Check for non-zero, non-negative helix length
    if l <= 0:
        raise ValueError( "Positive helix length required for DMI estimation if an initial DMI guess is not provided." )

    #Suggest an initial guess for d0 if one is not already there. This guess comes from the ipython notebook "ref-dmi-constant" without reference, but it is used here because it performs well in most examples.
    if d0 == None:
        d0 = 4 * np.pi * a / l

    #Find the range that d can exist in.
    dRange = Find_DMI_Signchange( a, ms, l, h, d0, tol * 1e2 )

    #Use an optimization routine to find d.
    def Helix_Length_Difference( d, a, ms, l, h ):
        return Find_Helix_Length( d, a, ms, h ) - l

    d = scipy.optimize.brentq( Helix_Length_Difference, dRange[0], dRange[1], args=( a, ms, l, h ), xtol=tol )

    #Cleanup and return
    if( verbose == False ):
        finmag.logger.setLevel( logLevel )
    return d

if __name__ == "__main__":

    # #Example material properties that work. This is expected to return d = 0.0002465, but gives 0.000242, which is pretty damn close.
    A = 3.53e-13                              #Isotropic exchange energy constant (J/m)
    Ms = 1.56e5                               #Magnetisation Saturaton (A/m)
    l = 22e-9                                 #Observed helix length (m)
    H = np.array( [ 1., 0., 0. ] ) * ms * 0.  #External magnetic field strength (A/m)
    D0 = 4 * np.pi * a / l                    #Dzyaloshinkii-Moriya exchange energy constant (J/m^2)

    lFound = Find_Helix_Length( D0, A, Ms, H )
    print( "Helix length given DMI: {:.2e} m.".format( lFound ) )

    #Material properties from the Leeds group.
    # A = 9.74e-14                   #Isotropic exchange energy constant (J/m)
    # Ms = 9.5e4                     #Magnetisation Saturation (A/m)
    # l = 11e-9                      #Observed helix length (m)
    # H = np.array( [ 0., 0., 0. ] ) #External magnetic field strength (A/m)

    #dFound = Find_DMI( a, ms, l, h ) #Yields D = 1.18e-4 J/m^2
    #print( "DMI given Helix length: {:.2e} J/m^2.".format( dFound ) )


