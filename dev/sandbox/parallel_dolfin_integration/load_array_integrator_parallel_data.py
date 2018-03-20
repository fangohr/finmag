import matplotlib.pyplot as plt
from dolfinh5tools import openh5

hFile = openh5("array_integrator_parallel", mode="r")
plt.plot(hFile.read(t=0, field_name="T").vector().array())
plt.plot(hFile.read(t=1, field_name="T").vector().array())
plt.show()
