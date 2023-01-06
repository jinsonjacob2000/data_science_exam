import numpy as np
from numpy.linalg import  inv

a=np.random.randint(10,size=(3,3))
print(a)

print("inverse of the matrix is ",np.linalg.inv(a))
print()

print("Rank of the matrix is ",np.linalg.matrix_rank(a))
print()

print("Determinant of the matrix is ",np.linalg.det(a))
print()
print("Flateen  to id aray ",np.ravel(a))
print()



e,v=np.linalg.eig(a)
print("The eigan value of the matrix is ",e)
print("The eigan vector of the matrix is ",v)