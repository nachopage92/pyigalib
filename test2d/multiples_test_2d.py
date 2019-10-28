#::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# Poisson problem
#
# (DESCRIBIR PROBLEMA)

def exact_sol(x,y,test):
	from numpy import exp

	if (test=='test_1'):
		A,b,xi,yi = constantes_poisson_test()
		usol=0.0 ; dudx=0.0 ; dudy=0.0
		for i in range(4):
			xd = x-xi[i]
			yd = y-yi[i]
			arg = -b[i]*(pow(xd,2)+pow(yd,2))
			coef = 2.0*A[i]*b[i]
			aux  = exp(arg)
			usol += A[i]*aux
			dudx += -coef*xd*aux
			dudy += -coef*yd*aux
		return usol,dudx,dudy

	elif (test=='test_2'):
		pi=np.pi
		exact=np.sin(pi*x)*np.sin(pi*y)
		dx = pi*np.cos(pi*x)*np.sin(pi*y) 
		dy = pi*np.sin(pi*x)*np.cos(pi*y) 
		return exact,dx,dy
	
	else:
		print('Error, caso no ingresado')
		return 

def rhs(x,y,test):
	from numpy import exp

	if (test == 'test_1'):
		rhs = 0.0
		A,b,xi,yi = constantes_poisson_test()
		for i in range(4):
			xd = x-xi[i]
			yd = y-yi[i]
			arg = -b[i]*( pow(xd,2) + pow(yd,2) )
			rhs += A[i]*b[i]*exp(arg)*( b[i]*pow(xd,2) + b[i]*pow(yd,2) - 1.0 )
			rhs = rhs*4
		return -rhs 

	elif (test == 'test_2'):
		pi=np.pi
		return 2.*pi**2*np.sin(pi*x)*np.sin(pi*y)

	else:
		print('Error, caso no ingresado')
		return 
		

def constantes_poisson_test():
	A  = [ 10.00 , 50.00 , 100.0 , 50.00  ]
	b  = [ 180.0 , 450.0 , 800.0 , 1000.0 ]
	xi = [ 0.510 , 0.310 , 0.730 , 0.280  ]
	yi = [ 0.520 , 0.340 , 0.710 , 0.720  ]
	return A,b,xi,yi 

#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::

def SC_pts(V,n,m,order):
	import numpy as np
	fi,dfi,ddfi = roots_of_fi(n)
	# select order
	if (order==0):
		roots = fi
	elif (order==1):
		roots = dfi
	elif (order==2):
		roots = ddfi
	else:
		print('orden superior a 2 no ingresado')
		return

	#out = np.zeros(m*len(roots))
	out = []
	c = 0
	print(roots)
	for j in range(p-1,m+1):
		for xi in roots:
			print(j,V[j])
			out.append( V[j] + 0.5*(V[j+1]-V[j])*(xi+1.) )
			c += 1
	return np.array(out)

def roots_of_fi(p):
	from numpy import sqrt

	if (p==1):
		root_fi   = [-1.,1.]
		root_dfi  = [ 0 ]
		root_ddfi = [   ] # NA

	elif (p==2):
		root_fi   = [-1.,0,1.]
		root_dfi  = [-1./sqrt(3.),1./sqrt(3.)]
		root_ddfi = [ 0. ]

	elif (p==3):
		root_fi   = [-sqrt(225.-30.*sqrt(30.))/15. , sqrt(225.-30.*sqrt(30.))/15. ]
		root_dfi  = [-1.,0.,1.]
		root_ddfi = [-1./sqrt(3.),1./sqrt(3.)]

	elif (p==4):
		root_fi   = [-1.,0.,1.]
		root_dfi  = [-sqrt(225.-30.*sqrt(30.))/15. , sqrt(225.-30.*sqrt(30.))/15. ]
		root_ddfi = [-1.,0.,1.]

	elif (p==5):
		root_fi   = [-0.5049185675126533,0.5049185675126533]
		root_dfi  = [-1.,0.,1.]
		root_ddfi = [-sqrt(225.-30.*sqrt(30.))/15. , sqrt(225.-30.*sqrt(30.))/15. ]

	elif (p==6):
		root_fi   = [-1.,0.,1.]
		root_dfi  = [-0.5049185675126533,0.5049185675126533]
		root_ddfi = [-1.,0.,1.]

	elif (p==7):
		root_fi  = [-0.503221894597504,0.503221894597504]
		root_dfi  = [-1.,0.,1.]
		root_ddfi = [-0.5049185675126533,0.5049185675126533]

	else:
		print('error, no implementado')
	
	return root_fi,root_dfi,root_ddfi


# Test selector ( 'test_1', 'test_2' )
test = 'test_2'




# p16 - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary
from pyigalib import *
from nurbs_factory import *
from math import log10
import numpy as np
from scipy.linalg import solve,inv
from scipy.interpolate import interp2d
from matplotlib import pyplot as plt

import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

from time import time

start = time()

### DEFINE APPROXIMATING B-SPLINE AND COLLOCATION MATRICES ###
# B-spline order:
p=3
# Index of the highest control point:
n=10
# Generate (at the moment only uniform clamped) knot vector (it must be clamped):
uvec = make_knot_vector(p,n+1,"clamped")
uvec = np.array(uvec)
#print uvec

#Scale knot vector to match the domain size:
if (test=='test_1'):
	a=0; b=1 # domain size in 1d
	uvec = uvec/float(uvec[p+n+1])
	# scale to [-1,1]
elif (test=='test_2'):
	a=-1; b=1 # domain size in 1d
	uvec = 2*uvec/float(uvec[p+n+1])-1. 
else:
	print('Error: caso no ingresado')

#print(uvec)
# Generate collocation points in 1D - we choose them to be Greville apscissae:
xcp = greville(uvec,p,n+1)
print()
print(':::: greville abs :::::')
print(xcp)
print(':::::::::::::::::::::::')
print()

xcp = SC_pts(uvec,p,n+1,2)
print()
print(':::: superconvergent :::::')
print(xcp)
print(':::::::::::::::::::::::')
print()

input()

# Tensor product grid
ycp=xcp  
x,y = np.meshgrid(xcp[1:n], ycp[1:n])
x=x.flatten(1)
y=y.flatten(1)

# B-spline basis function and derivatives upt to 2nd order:
ders = basis_funs_and_derivatives_cpt(p,n,uvec,xcp,2)
# Matrix N_{i,j}==N_{j,p}(xcp(i)), (i=0,n+1,j=0,n+1) of basis functions, each row for each collocation point, each column for each basis fun.
N = ders[:,0,:]
# Matrix D2_{i,j}==d^2 N_{j,p}(xcp(i)) / dx^2, (i=0,n+1,j=0,n+1) of 2nd derivatives of basis functions, each row for each collocation point, each column for each basis fun.
D2 = ders[:,2,:]

### BOUNDARY CONDITIONS ###
# Apply boundary conditions (Homogenous Dirichlet):
N = N[1:n,1:n]
D2 = D2[1:n,1:n]

# RHS vector
f = rhs(x,y,test)

# PDE operator-tensor product Laplacian
L=-(np.kron(N,D2)+np.kron(D2,N)) #poisson

# condition number
print (p,n+1,np.linalg.cond(L))
##sparsity pattern of L
#fig=plt.figure()
#plt.title('Sparsity pattern of discretization matrix for Laplace operator, p = %d, n = %d'  % (p,n))
#plt.spy(L, precision=1e-50, marker='s', markersize=2)
#plt.show()

# Solve system
u=solve(L,f) 
elapsed = (time() - start)

# Reshape long 1D results to 2D grid:
Py=np.zeros((n+1,n+1))
Py[1:n,1:n] = u.reshape(n-1,n-1) # ver np.kron para entender este paso

# Interpolate to finer grid for plotting
nsamples = 50
xnd = np.linspace(a,b,nsamples)  # equidistant grid
ynd=xnd
xx,yy = np.meshgrid(xnd,ynd)
xx=xx.flatten(1)
yy=yy.flatten(1)

# Basis functions matrix for sampled points
N = basis_funs_cpt(p,n,uvec,xnd)

# Napravi nurbs povrs:
uu=np.zeros((nsamples,nsamples))
# Sustina B-spline interpolacije u jednoj komandi
uu = np.dot(np.kron(N,N),Py.flatten(1))
exact,exact_dudx,exact_dudy = exact_sol(xx,yy,test)

# Exact solution and Error
maxerr=max(abs(uu-exact))
#print p,n,maxerr,elapsed

# Prepare for plotting
uu=np.reshape(uu,(nsamples,nsamples))
xx=np.reshape(xx,(nsamples,nsamples))
yy=np.reshape(yy,(nsamples,nsamples))
# For control polygon:
x,y = np.meshgrid(xcp, ycp)

# Plot results
fig=plt.figure()
ax = p3.Axes3D(fig)
ax.plot_wireframe(x,y,Py,color='black', alpha=0.4)
ax.plot_surface(xx, yy, uu, rstride=1, cstride=1, cmap=cm.cool, linewidth=0.3, antialiased=True, alpha=0.7)
ax.text(-2.3,1.58,0.5,'p = %d, n = %d, (max err) = %e, time = %f [s]' % (p,n+1,maxerr,elapsed))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()

## If you have it installed:
#from mayavi import mlab
#mlab.surf(xnd,ynd,uu)
#mlab.show()
#mlab.savefig('poisson-mlab.png')
