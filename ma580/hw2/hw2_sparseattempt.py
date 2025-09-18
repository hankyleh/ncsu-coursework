# hw 2 computer problem
# finite diff with sparse matrix storage

import numpy
from numpy import linalg as linalg
from numpy import matrix as matrix
import matplotlib
from matplotlib import pyplot as plt
import time
import random as rng

class sparse_mat:
    def __init__(self, v, c, r):
            self.val = v
            self.row = r
            self.col = c
    def encode(A):
        dim = A.shape
        r = numpy.zeros(dim[0]+1, dtype=int)
        v = numpy.zeros(1)
        c = numpy.zeros(1)

        col_array = numpy.arange(0, dim[1])
        print(col_array)
        for i in range(0, dim[0]):
            r[i+1] = r[i] + sum(1*(A[i]!=0))
            v = numpy.append(v, A[i][A[i]!=0])
            c = numpy.append(c, col_array[A[i]!=0])

        v = v[1:]
        c = c[1:]
        return sparse_mat(v,c,r)
    def revert(self):
        pass
    def access(self, i, j):
        mask = (self.col[self.row[i]:self.row[i+1]]==j)
        val = self.val[self.row[i]:self.row[i+1]] @ matrix.transpose(mask)
        return val
    

# generate random nxn sparse matrix
n = 1000

A = numpy.zeros((n,n))
for i in range(0,n):
    A[i][i] = rng.random()
    for x in numpy.arange(int(numpy.floor(n/15)+1)):
        j = rng.randint(0, n-1)
        A[i][j] = rng.random()

print(A)
B = sparse_mat.encode(A)



for x in range(0, 10):
    i = rng.randint(0,n-1)
    j = rng.randint(0,n-1)

    print("index: ("+str(i)+","+str(j)+")")
    print(A[i][j])
    print(B.access(i,j))


# speed test

ntests = 10000
i = numpy.zeros(ntests, dtype=int)
j = numpy.zeros(ntests, dtype=int)

for x in range(0, ntests):
    i[x] = rng.randint(0,n-1)
    j[x] = rng.randint(0,n-1)
b = 0
print("running original case...")
start = time.time_ns()
for x in range(0, ntests):
    b = A[i[x]][j[x]]
running_og = (time.time_ns()) - (start)

print("running sparse storage case...")
start = time.time_ns()
for x in range(0, ntests):
    b = B.access(i[x],j[x])
running_sparse = (time.time_ns()) - (start)

print("original method")
print("     "+str(running_og/ntests))
print("sparse method")
print("     "+str(running_sparse/ntests))

# plt.plot()
# plt.spy(A)
# plt.show()
# plt.close()





# part a -- coefficients

# part b -- solver

# part c -- condition numbers

# part d -- plotting