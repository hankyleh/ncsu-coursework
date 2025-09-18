import numpy
import scipy
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
    # I = sparse.eye(A.shape[0], A.shape[1])
    # A_inv = sparse.linalg.spsolve(A, I)
    # cond[ni] = sparse.linalg.onenormest(A)*sparse.linalg.onenormest(A_inv)
    if (ni>0):
        r[ni] = e[ni]/e[ni-1]
    if n[ni] == 2**10:
        x, y = numpy.meshgrid(x, y)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(x, y,(U-K), cmap=matplotlib.cm.Reds)
        plt.title("Absolute Error, h="+str(h[ni]))
        plt.show()

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(x, y,U, cmap=matplotlib.cm.Blues)
        plt.title("Numerical solution, h="+str(h[ni]))
        plt.show()

print("h")
print(h)
print("n")
print(n)
print("e")
print(e)
print("r")
print(r)
print("condition number")
print(cond)