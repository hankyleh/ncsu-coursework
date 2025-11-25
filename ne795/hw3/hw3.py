import numpy
import scipy
import scipy.integrate as integrate
import matplotlib
from matplotlib import pyplot as plt

global h, c, k, sigR, aR
h = numpy.float128(4.135667696e-15) # eV * s
k = numpy.float128(8.617333262e-5)  # eV / K
c = numpy.float128(2.99792458e10)   # cm / s
sigR = numpy.float128(3.53916934e7) # eV / (cm^2 * s * K^4)
aR = numpy.float128(4.72215928e-3)  # eV / (cm^2 * s * K^4) 


def planck(T, nu):
    B = 2*h*(nu**3)/((c**2)*(numpy.exp((h*nu)/(k*T)) - 1))
    return(B)

def int_planck(T, nu):
    z = (h*nu)/(k*T)
    if (z <= 2):
        sigma = (z**3) * ((1/3) - (z/8) + ((z**2)/62.4))
    else:
        sigma = 6.4939 - (numpy.exp(-z) * ((z**3)+(3*(z**2))+(6*z)+7.28))
    return(sigma)

def FC_opacity(k0, T, nu):
    kappa = (k0/(h*nu)**3)*(1-numpy.exp(-h*nu/(k*T)))
    return kappa

def group_opacity_planck(T, nu, kappa):
    z0 = (h*nu[0])/(k*T)
    z1 = (h*nu[1])/(k*T)
    sigma = (z0**3)+(3*z0**2)+(6*z0)+7.28 - (numpy.exp(z0-z1)*((z1**3)+(3*z1**2)+(6*z1)+7.28))
    int_k = kappa * (1 - numpy.exp(z0-z1)) / (sigma* (k*T)**3)
    return int_k

def dBdT(T, nu):
    d = ((2* h**2 * nu**4)/(c**2 * k * T**2))*numpy.exp(h*nu/(k*T))/((numpy.exp(h*nu/(k*T)) - 1)**2)
    return d

def fr(T, nu):
    return((h*nu)/(k*T))

def group_opacity_rosseland(T, nu, kappa):
    def num_func(n):
        return (h**3)*(n**7)*(1-numpy.exp(-h*n/(k*T)))**-2
    def dem_func(n):
        return (n**4)*(1-numpy.exp(-h*n/(k*T)))**-2

    numerator, err = integrate.quad(num_func, nu[0], nu[1])
    denominator, err = integrate.quad(dem_func, nu[0], nu[1])
    return 27* denominator/numerator




T = numpy.array([1, 10**2, 10**3]/k).astype(numpy.float128)
nu = numpy.zeros((3, 1000))

nu[0] = numpy.linspace(0.001, 1e1, 1000)*(1/h)
nu[1] = numpy.linspace(0.001, 1e3, 1000)*(1/h)
nu[2] = numpy.linspace(0.001, 1e4, 1000)*(1/h)


colors = ['green', 'orange', 'red']
for t in range(0, len(T)):
    plt.figure(dpi=350)
    plt.plot(h*nu[t], planck(T[t], (nu[t])), label=f"kT = {k*T[t]:.1f} eV", color=colors[t])
    plt.legend()
    plt.title(f"Planck Function, kT = {k*T[t]:.1f} eV")
    plt.xlabel("$h\\nu$ [eV]")
    plt.ylabel("$B_\\nu(T)$ [eV/cm2]")
    # plt.show()
    plt.savefig(f"planck{T[t]*k:.1f}.png")
    plt.close()

bounds = numpy.array([0.000, 0.7075, 1.415, 2.123, 2.830, 3.538, 4.245,
    5.129, 6.014, 6.898, 7.783, 8.667, 9.551, 10.44, 11.32, 12.20, 
    13.09, 1e4])*(1000/h)

bin_widths = (bounds[1:] - bounds[0:-1])
log_nu = numpy.linspace(numpy.log(0.001*bounds[1]), numpy.log(bounds[-1]), 250)
nu = numpy.exp(log_nu)

B_g = numpy.zeros((3, len(bounds)-1))
for t in range(0, len(T)):
    for g in range(0, len(bounds)-1):
        B_g[t][g] = (2*(k*T[t])**4)*(int_planck(T[t], bounds[g+1]) - int_planck(T[t], bounds[g]))/((h**3)*(c**2))
    plt.figure(dpi=350)
    plt.plot(nu*h, planck(T[t], nu), label= "Planck Function")
    plt.stairs((B_g[t]/(bin_widths)), edges = (bounds*h), label="Group approximation")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-8, 1e20)
    plt.legend()
    plt.title(f"Group Planck Aprpoximations, T={T[t]*k:.1f} eV")
    plt.xlabel("$h\\nu$ [eV]")
    plt.ylabel("$B(\\nu)$")
    # plt.show()
    plt.savefig(f"group_planck{T[t]*k:.1f}.png")
    plt.close()

kappa_B_g  = numpy.zeros(B_g.shape)
kappa_R_g  = numpy.zeros(B_g.shape)
for t in range(0, len(T)):
    for g in range(0, len(bounds)-1):
        z = fr(T[t], bounds[g])
        z1 = fr(T[t], bounds[g+1])
        kappa_B_g[t][g] = group_opacity_planck(T[t], bounds[g:g+2], 27)
        kappa_R_g[t][g] = group_opacity_rosseland(T[t], bounds[g:g+2], FC_opacity)

    plt.figure(dpi=350)
    plt.plot(nu*h, FC_opacity(27, T[t], nu), label = "Spectral Opacity")
    plt.stairs(kappa_B_g[t], edges = (bounds*h), label = "Group Opacity")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Planck Group Opacities, T={T[t]*k:.1f} eV")
    plt.xlabel("$h\\nu$ [eV]")
    plt.ylabel("$\\varkappa$")
    # plt.show()
    plt.savefig(f"planckopacities{T[t]*k:.1f}.png")
    plt.close()

    plt.figure(dpi=350)
    plt.plot(nu*h, FC_opacity(27, T[t], nu), label = "Spectral Opacity")
    plt.stairs(kappa_R_g[t], edges = (bounds*h), label = "Group Opacity")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Rosseland Group Opacities, T={T[t]*k:.1f} eV")
    plt.xlabel("$h\\nu$ [eV]")
    plt.ylabel("$\\varkappa$")
    # plt.show()
    plt.savefig(f"rosseland{T[t]*k:.1f}.png")
    plt.close()


    plt.figure(dpi=350)
    plt.stairs(kappa_R_g[t], edges = (bounds*h), label = "Rosseland")
    plt.stairs(kappa_B_g[t], edges = (bounds*h), label = "Planck")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$h\\nu$ [eV]")
    plt.ylabel("$\\varkappa$")
    plt.title(f"Group Opacity Comparison, T = {T[t]*k:.1f} [eV]")
    plt.savefig(f"comparison{T[t]*k:.1f}.png")
    plt.close()




    print(kappa_B_g[t])
    print(kappa_R_g[t])
