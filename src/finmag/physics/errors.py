class UnknownInteraction(KeyError):
    def __init__(self, unknown_interaction, known_interactions):
        self.unknown = unknown_interaction
        self.known = known_interactions
        super(UnknownInteraction, self).__init__()

    def __str__(self):
        message = ("Couldn't find interaction with name {}. Do you mean "
                "one of {}?".format(self.unknown, ", ".join(self.known)))
        return message


