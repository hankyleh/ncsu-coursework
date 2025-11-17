
import numpy
import numpy.matlib as matlib
# import matplotlib
import matplotlib.pyplot as plt

def continuous_eig (t, c, l):
    first = (c/2)/(1-c+(1/(3*t))*l**2)
    second = ((-6*l) + 2*(3+l**2)*numpy.arctan(l))/(3*l)
    return first*second

def discrete_eig (l, st, c, dx, mu, w):
    sa = st*(1-c)
    tau = st*dx/mu
    alpha = (1/tau) - 1/(numpy.exp(tau)-1)
    beta = 2*alpha - 1
    a = 1/(3*st*dx)


    omega = 0*l

    for m in range(len(mu)):
        print(m)
        omega = omega + (
            (
                w[m]*c*((1/3)-mu[m]**2)*(2-2*numpy.cos(l*st*dx))*(numpy.exp(0.5j*l*st*dx) - 2j*alpha[m]*numpy.sin(0.5*l*st*dx))
            )/(
                2*st*dx*(
                    (
                        ((2/3)+(st**2)*(1-c)*dx**2)-((2/3)*numpy.cos(l*st*dx))
                    )*(
                        ((1/(numpy.exp(tau[m])-1))*2j*numpy.sin(0.5*l*st*dx))+numpy.exp(0.5j*l*st*dx)
                    )
                )
            )
        )
    omega = numpy.abs(omega)
    # print(omega)
    # print(l)
    return omega


# Continuous Fourier Modes
c_vec = [0.1, 0.5, 0.9, 0.999, 1]
l_vec = numpy.linspace(0.001, 20, 300)



c_vec_p2 = [0.5, 0.9, 1]
dx_vec = [0.01, 0.1]

[mu, w] = numpy.polynomial.legendre.leggauss(8)
print(mu)
print(w)



continuous_omega = numpy.zeros((len(c_vec), len(l_vec)))
discrete_omega = numpy.zeros((len(c_vec), len(l_vec)))

sig_tot = 1.0



plt.figure
for c in range(len(c_vec)):
    continuous_omega[c] = continuous_eig(sig_tot, c_vec[c], l_vec)
    plt.plot(l_vec, continuous_omega[c], label=f"$c = {c_vec[c]}$")

spec_rad = numpy.max(continuous_omega, axis=1, keepdims=True)
spec_eig = ((matlib.repmat(l_vec, len(c_vec), 1))[continuous_omega==spec_rad])

print(spec_rad)
print(spec_eig)

plt.scatter(spec_eig, spec_rad, s=15)
plt.legend()
plt.title("Continuous Form Fourier Modes")
plt.xlabel("$\\lambda$")
plt.ylabel("$|\\omega$|")
# plt.show()
plt.close()


for c in c_vec_p2:
    plt.figure
    for dx in dx_vec:
        # bound = numpy.pi * dx * sig_tot
        l_vec_p2 = numpy.linspace(0.001, 80, 1000)
        plt.plot(l_vec_p2, discrete_eig(l_vec_p2, sig_tot, c, dx, mu, w), label=f"{dx}")
    plt.legend()
    # plt.yscale("log")
    plt.show()
    plt.close()



print(sum(w))




# Discretized Fourier Modes