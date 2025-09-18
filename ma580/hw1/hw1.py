import numpy
import matplotlib.pyplot as plt

class solution:
    def __init__(self, x, u):
            self.mesh = x
            self.values = u

pi = numpy.pi

def sourceFunc(x):
      return 2*(pi**2)*((17*(numpy.cos(3*pi*x))*(numpy.sin(5*pi*x)))  
                        + (15*(numpy.sin(3*pi*x))*(numpy.cos(5*pi*x))))

def analyticSolution(x):
      return numpy.cos(3*pi*x)*numpy.sin(5*pi*x)
      
def elliptic_solve_1d(f,n):
       h = 1/n
       x = numpy.linspace(0, 1, n+1)

       b = f(x)
       b[0] = 0
       b[n] = 0

       A = numpy.zeros((n+1, n+1))
       A[0][0] = 1
       A[n][n] = 1
       
       for i in range(1, n):
             A[i][i] = 2/(h**2)
             A[i][i-1] = -1/(h**2)
             A[i][i+1] = -1/(h**2)
       u = numpy.linalg.inv(A) @ b
       return solution(x, u)



# 
# Plot results
# 


mesh_analytic = numpy.linspace(0, 1, 200)
soln_analytic = analyticSolution(mesh_analytic)

k = numpy.linspace(3, 10, 8).astype('int')
print(k)
n = (2**k).astype('int')

# All values of k
plt.figure(dpi=250)
for i in n:
       s = elliptic_solve_1d(sourceFunc,i)
       plt.plot(s.mesh, s.values, label=i)

plt.plot(mesh_analytic, soln_analytic, "--", label="Analytic")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("All numerical Solutions")
# plt.savefig("all_num_solns")
# plt.show()
plt.close()

# Least k
plt.figure(dpi=250)
s = elliptic_solve_1d(sourceFunc,8)
plt.plot(s.mesh, s.values, label="Numerical")
plt.plot(mesh_analytic, soln_analytic, "--", label="Analytic")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("n=8")
# plt.savefig("lowk")
# plt.show()
plt.close()

# Greatest k
plt.figure(dpi=250)
s = elliptic_solve_1d(sourceFunc,1024)
plt.plot(s.mesh, s.values, label="Numerical")
plt.plot(mesh_analytic, soln_analytic, "--", label="Analytic")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("n=1024")
# plt.savefig("highk")
# plt.show()
plt.close()

# Intermediate k
plt.figure(dpi=250)
s = elliptic_solve_1d(sourceFunc,128)
plt.plot(s.mesh, s.values, label="Numerical")
plt.plot(mesh_analytic, soln_analytic, "--", label="Analytic")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("n=128")
# plt.savefig("midk")
# plt.show()
plt.close()



E = numpy.zeros(k.shape)

for i in range(n.size):
      s = elliptic_solve_1d(sourceFunc, n[i])
      E[i] = max(abs(analyticSolution(s.mesh) - s.values))

plt.figure()
plt.plot(k, E)
plt.yscale('log')
plt.title("L-inf absolute error")
plt.ylabel("E")
plt.xlabel("k")
# plt.savefig("abs_err")
# plt.show()
plt.close()

E_ratio = E[1:] / E[:-1]
print(E[1:])
print(E[:-1])
print(E_ratio)

plt.figure()
plt.scatter(k[1:], E_ratio, label="E_k / E_k-1")
plt.hlines(0.25, k[0], k[-1], color="black", linestyle="--",  label="0.25")
plt.xlabel("k")
plt.title("Error Ratio")
plt.legend()
# plt.savefig("convergence")
# plt.show()
plt.close()

