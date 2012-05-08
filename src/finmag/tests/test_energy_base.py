from finmag.energies.energy_base import EnergyBase

"""
Shows how to conform to the abstract base class
for the energies through subclassing.

"""

class MyEnergy(EnergyBase):
    def __init__(self, myConstant1, myConstant2):
        """
        The __init__ method is not mandated by
        the abstract base class and can take
        any parameters needed.

        """
        pass

    def setup(self, S3, m, Ms, unit_length):
        """
        This method is required.

        The setup will most likely be called
        by the new Sim class. We should agree
        on a common set of parameters for all
        energies.

        """
        pass

    def compute_field(self):
        """
        This method is required.

        We could think about passing t to
        compute_field for time-dependent
        external fields, or m, if we want
        to refactor the whole field computation
        process.

        """
        pass

    def compute_energy(self):
        """
        This method is required.

        """
        pass

def test_my_energy():
    try:
        my_energy = MyEnergy(1, 2)
        assert True
    except Exception, ex:
        print "Problem initialising MyEnergy."
        print ex
        assert False
