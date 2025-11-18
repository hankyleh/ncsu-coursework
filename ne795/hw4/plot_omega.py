
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
    alph_p = 1 - 2*alpha
    beta = 0.5*st*dx*l
    a = 1/(3*st*dx)
    omega = 0*l

    for m in range(len(mu)):
        # print(m)
        omega = omega + (
            (
                (c/2)*w[m]*((1/3)-mu[m]**2) *(1 + alph_p[m]*((2*mu[m]/(st*dx))+alph_p[m])* (numpy.tan(beta))**2)*(4*(numpy.sin(beta)**2))
            )/(
                (1 + (((2*mu[m]/(st*dx))+alph_p[m])**2)*(numpy.tan(beta)**2)) * (sa*dx + 4*a*(numpy.sin(beta)**2))*(st*dx)
            )
        )
    omega = numpy.abs(omega)
    # print(omega)
    # print(l)
    return omega


# Continuous Fourier Modes
c_vec = [0.1, 0.5, 0.9, 0.999, 1]
l_vec = numpy.linspace(0.001, 25, 600)



c_vec_p2 = [0.5, 0.9, 1]
l_vec_p2 = numpy.linspace(0.001, 25, 600)
dx_vec = [0.01, 0.1, 1, 10, 100]

[mu, w] = numpy.polynomial.legendre.leggauss(8)
print(mu)
print(w)



continuous_omega = numpy.zeros((len(c_vec), len(l_vec)))
discrete_omega = numpy.zeros((len(c_vec_p2), len(dx_vec), len(l_vec_p2)))

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
plt.title("Continuous Fourier Eigenvalues")
plt.savefig("continuous.png")
# plt.show()
plt.close()

spec_rad = numpy.zeros((len(c_vec_p2), len(dx_vec)))
spec_eig = numpy.zeros((len(c_vec_p2), len(dx_vec)))


for c in range(len(c_vec_p2)):
    plt.figure
    for dx in range(len(dx_vec)):
        # bound = numpy.pi * dx * sig_tot
        discrete_omega[c][dx] = discrete_eig(l_vec_p2, sig_tot, c_vec_p2[c], dx_vec[dx], mu, w)
        plt.plot(l_vec_p2, discrete_eig(l_vec_p2, sig_tot, c_vec_p2[c], dx_vec[dx], mu, w), label=f"dx = {dx_vec[dx]}")
        spec_rad[c][dx] = numpy.max(discrete_omega[c][dx])
        spec_eig[c][dx] = l_vec_p2[discrete_omega[c][dx]==spec_rad[c][dx]]
        plt.scatter(spec_eig[c][dx], spec_rad[c][dx], s=15)
    print(f"for c={c_vec_p2[c]}")
    print(spec_rad[c])
    print(spec_eig[c])
    plt.legend()
    # plt.yscale("log")
    plt.xlabel("$\\lambda$")
    plt.ylabel("$|\\omega$|")
    plt.title(f"SC, SN Discretized Fourier Eigenvalues, c={c_vec_p2[c]}")
    plt.savefig(f"disc_{c_vec_p2[c]}.png")
    plt.show()
    plt.close()






# Discretized Fourier Modes