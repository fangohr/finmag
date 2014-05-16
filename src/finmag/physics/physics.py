"""
Derives physical quantities from the primary simulation state.

"""
import logging

logger = logging.getLogger(name="finmag")


class Physics(object):
    def __init__(self, S1, S3, m, Ms, unit_length, equation="llg"):
        if equation == "llg":
            from finmag.physics.llg import LLG
            self._eq = LLG(S1, S3, do_precession=True, average=False, unit_length=unit_length)
        elif equation == "sllg":
            from finmag.physics.llb.sllg import SLLG
            self._eq = SLLG(S1, S3, unit_length=unit_length)
        elif equation == "llg_stt":
            from finmag.physics.llg_stt import LLG_STT
            self._eq = LLG_STT(S1, S3, unit_length=unit_length)
        else:
            raise ValueError("Equation must be one llg, sllg or llg_stt.")

    def add(self, interaction, with_time_update=None):
        self._eq.effective_field.add(interaction, with_time_update)

    def interaction(self, interaction_name):
        return self._eq.effective_field.get(interaction_name)

    def interactions(self):
        return self._eq.effective_field.all()

    def remove(self, interaction_name):
        return self._eq.effective_field.remove(interaction_name)

    def includes(self, interaction_name):
        return self._eq.effective_field.exists(interaction_name)

    def effective_field(self):
        return self._eq.effective_field.compute()

    def energy(self, interaction_name=None):
        if interaction_name is None:
            return self._eq.effective_field.total_energy()
        return self.interaction(interaction_name).compute_energy()

    @property
    def do_precession(self):
        return self._eq.do_precession

    @do_precession.setter
    def do_precession(self, value):
        self._eq.do_precession = value


