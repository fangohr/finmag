import finmag.native as n

print n.llg.demo_hello("x")


def test_demo_hello():
    assert n.llg.demo_hello("x") == "Hellox"
    assert n.llg.demo_hello("y z 1") == "Helloy z 1"
