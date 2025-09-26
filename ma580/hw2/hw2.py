import numpy
# import scipy
import scipy.sparse as sparse
import matplotlib
import matplotlib.pyplot as plt

pi = numpy.pi
# part a -- matrix assignment
def source_function(x, y):
    z = -2*(pi**2)*((numpy.cos(pi*x)**2 * numpy.sin(2*pi*y)**2) 
                    -5*( numpy.sin(pi*x)**2 * numpy.sin(2*pi*y)**2) 
                    +4*(numpy.sin(pi*x)**2 * numpy.cos(2*pi*y)**2))
    return z

def analytic_solution(x, y):
    return numpy.sin(pi*x)**2 * numpy.sin(2*pi*y)**2
    
class poisson_system:
    def __init__(self, x, y, A, f):
        self.x_mesh = x
        self.y_mesh = y
        self.coefficients = A
        self.source = f

def unpack_1d_array(a):
    m = int(numpy.sqrt(a.size))
    n = m+1
    A = numpy.zeros((m,m))
    for i in range(0, m):
        for j in range(0, m):
            A[i][j] = a[i+ (j)*(m)]
    return A

def system_def(f, n):
    m = n-1
    I = sparse.eye(m,m)
    e = numpy.ones(m)
    T = sparse.diags([-e, 4*e, -e], [-1, 0, 1], (m,m))
    S = sparse.diags([-e, -e], [-1, 1], (m,m))
    A = sparse.kron(I, T) + sparse.kron(S, I)
    
    h = 1/n
    x = numpy.arange(h, 1, h)
    y = numpy.arange(h, 1, h)
    q_vec = numpy.zeros((n-1)**2)

    for i in range(0, (n-1)):
        for j in range(0, (n-1)):
            k = i+ (j)*(n-1)
            q_vec[k] = (h**2)*source_function(x[i], y[j])

    return poisson_system(x, y, A, q_vec)

def hager_invnorm(A):
    # Hager 1984, Higham 1988
    n = A.shape[0]
    x = (1/n)*numpy.ones((n))
    while (True):
        y = sparse.linalg.spsolve(A, x)
        # heaviside instead of sign() so that sign(0) = 1
        xi = 2*numpy.heaviside(y, 1) - 1 
        z = sparse.linalg.spsolve(A,xi)
        if (numpy.max(abs(z)) <= z@x):
            return numpy.sum(abs(y))
        x = numpy.zeros(n)
        x[numpy.argmax(z)] = 1


k = numpy.arange(3, 11, 1)
n = 2**k
h = 1/n
e = numpy.zeros(n.shape)
r = e.copy()
cond = e.copy()

for ni in range(0, n.size):
    print("n= "+str(n[ni]))
    system = system_def(source_function, n[ni])
    A = system.coefficients
    x = system.x_mesh
    y = system.y_mesh
    q = system.source

    u = sparse.linalg.spsolve(A, q)
    U = unpack_1d_array(u)
 
    K = numpy.zeros(U.shape)
    for i in range(0, K.shape[0]):
        for j in range(0, K.shape[1]):
            K[i][j] = analytic_solution(x[i], y[j])

    e[ni] = numpy.max(numpy.abs(K-U))
    cond[ni] = sparse.linalg.onenormest(A)*hager_invnorm(A)
    print(cond[ni])
    if (ni>0):
        r[ni] = e[ni]/e[ni-1]
    if n[ni] == 2**10:
        x, y = numpy.meshgrid(x, y)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(x, y,(U-K), cmap=matplotlib.cm.Reds)
        plt.title("Absolute Error, $h="+str(h[ni])+"$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$f_c - f$")
        plt.show()
        # plt.savefig("abs_errs")
        plt.close()

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(x, y,U, cmap=matplotlib.cm.Blues)
        plt.title("Numerical solution, $h="+str(h[ni])+"$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$f_c(x,y)$")
        plt.show()
        # plt.savefig("num_soln")
        plt.close()

fig = plt.figure()
plt.scatter(n, cond)
plt.title("Condition Number Estimates")
plt.xlabel("$n$")
plt.ylabel("$\\kappa_1(A)$, upper bound")
plt.show()
# plt.savefig("condition_linear")
plt.xscale("log")
plt.yscale("log")
plt.show()
# plt.savefig("condition_log")
plt.close()
