import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy.matlib as matlib

def continuous_eig (t, c, l):
    first = (c/2)/(1-c+(1/(3*t))*l**2)
    second = ((-6*l) + 2*(3+l**2)*numpy.arctan(l))/(3*l)
    return first*second


# Continuous Fourier Modes
c_vec = [0.1, 0.5, 0.9, 0.999, 1]
l_vec = numpy.linspace(0.001, 20, 50)

continuous_omega = numpy.zeros((len(c_vec), len(l_vec)))


plt.figure
for c in range(len(c_vec)):
    continuous_omega[c] = continuous_eig(1, c_vec[c], l_vec)
    plt.plot(l_vec, continuous_omega[c], label=f"$c = {c_vec[c]}$")

spec_rad = numpy.max(continuous_omega, axis=1, keepdims=True)
spec_eig = ((matlib.repmat(l_vec, len(c_vec), 1))[continuous_omega==spec_rad])

plt.scatter(spec_eig, spec_rad, s=15)
plt.legend()
plt.title("Continuous Form Fourier Modes")
plt.xlabel("$\\lambda$")
plt.ylabel("$|\\omega$|")
plt.show()
plt.close()



# Discretized Fourier Modes