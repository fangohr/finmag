class Simulation(object):
    max_instance_counter = 0
    instances = {}

    def __init__(self):
        self.id = Simulation.max_instance_counter
        Simulation.max_instance_counter += 1
        assert self.id not in Simulation.instances.keys() 
        Simulation.instances[self.id] = self

    def __str__(self):
        return "{} / {}".format(self.id, Simulation.max_instance_counter)

    def shutdown(self):
        """Remove all (cyclic) references to this object - should only
        be called in preparation of getting the object going out of 
        scope and  being garbage collected."""

        assert self.id in Simulation.instances.keys()
        del Simulation.instances[self.id]

    def __del__(self):
        print("Instance {} is garbage collected".format(self.id))


    def instances_list_all(self):
        for id_ in sorted(Simulation.instances.keys()):
            if id_ != None:   # can happen if instances have been deleted
                print("sim {}: {}".format(id_, Simulation.instances[id_]))

    def instances_delete_all_others(self):
        for id_ in sorted(Simulation.instances.keys()):
            if id_ != None:   # can happen if instances have been deleted
                if id_ != self.id:  # do not delete ourselves, here
                    sim = Simulation.instances[id_]
                    sim.shutdown()
                    del sim

    def instances_in_memory_count(self):
        return sum([1 for id_ in Simulation.instances.keys() if id_ != None])


def test_shutdown1():

    a = Simulation()
    assert a.instances_in_memory_count() == 1
    print(a)
    a.shutdown()
    assert a.instances_in_memory_count() == 0
    del a

def test_shutdown2():

    a = Simulation()
    assert a.instances_in_memory_count() == 1
    print(a)
    b = Simulation()
    assert a.instances_in_memory_count() == 2
    a.shutdown()
    assert a.instances_in_memory_count() == 1
    b.shutdown()
    assert a.instances_in_memory_count() == 0
    assert Simulation.max_instance_counter == 2 + 1

    del a
    assert Simulation.max_instance_counter == 2 + 1

    del b

    assert Simulation.max_instance_counter == 2 + 1


if __name__ == "__main__":

    test_shutdown1()
    test_shutdown2()

def demo():
    b = Simulation()
    print(b)
    del b
    c = Simulation()
    print(c)
    print("Instances alive: {}".format(c.instances_in_memory_count()))
    c.instances_list_all()
    c.instances_delete_all_others()
    print("Instances alive: {}".format(c.instances_in_memory_count()))
    c.instances_list_all()
