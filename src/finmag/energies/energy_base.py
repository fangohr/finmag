import abc

class EnergyBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def setup(S3, m, unit_length):
        return

    @abc.abstractmethod
    def compute_field(self):
        return

    @abc.abstractmethod
    def compute_energy(self):
        return
