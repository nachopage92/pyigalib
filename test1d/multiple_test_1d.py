#
# Solving two-point BVP: -u''+u'+u = f, f= (1+4\pi^2)*sin(2\pi x)-2*\pi*cos(2\pi x), x \in [0,1], 
# Exact solution: u  = sin(2\pi x)
# Boundary conditions are homogenous Dirichlet u[0]=0, u[1]=0.
#

# Test selector
tests = ['RW','T2','Waveprop','Axialtruss']
test=tests[1]


# ----------------------------------------------------------------------------------------------


import numpy as np
from pyigalib import *
from nurbs_factory import *
from scipy.linalg import solve
from math import log10
import matplotlib.pyplot as plt

### DEFINE APPROXIMATING B-SPLINE AND COLLOCATION MATRICES ###
# B-spline order:
p=2
# Index of the highest control point:
n=100 
# Generate (at the moment only uniform clamped) knot vector (it must be clamped):
uvec = make_knot_vector(p,n+1,"clamped")
#Scale knot vectro to match the domain size:
uvec = np.array(uvec)
uvec = uvec/float(uvec[p+n+1])
# Generate collocation points in 1D - we choose them to be Greville abscissae:
xcp = greville(uvec,p,n+1)

# B-spline basis function and derivatives upt to 2nd order:
ders = basis_funs_and_derivatives_cpt(p,n,uvec,xcp,2)
# Matrix N_{i,j}==N_{j,p}(xcp(i)), (i=0,n+1,j=0,n+1) of basis functions, each row for each collocation point, each column for each basis fun.
N = ders[:,0,:]
# Matrix D2_{i,j}==d N_{j,p}(xcp(i)) / dx, (i=0,n+1,j=0,n+1) of first derivatives of basis functions, each row for each collocation point, each column for each basis fun.
D = ders[:,1,:]
# Matrix D2_{i,j}==d^2 N_{j,p}(xcp(i)) / dx^2, (i=0,n+1,j=0,n+1) of 2nd derivatives of basis functions, each row for each collocation point, each column for each basis fun.
D2 = ders[:,2,:]

### BOUNDARY CONDITIONS ###
# Create RHS vector for the given problem:
def exact_sol(test,x,**kwargs):
	#------------------------------------------
	# entrega la solucion exacta y su derivada
	#------------------------------------------
	import numpy as np
	if (test=='RW'):
		xo = kwargs.get('xo',0.4)
		a = kwargs.get('alpha',50.0)
		u  = (1.-x)*(np.arctan(a*(x-xo))+np.arctan(a*xo))
		du = (1.-x)*a/(1+a*a*(x-xo)*(x-xo))-(np.arctan(a*(x-xo))+np.arctan(a*xo))
	elif (test=='T2'):
		u  = 1. - pow(x,3) + np.exp(-100.*pow(x,2))
		du = -3.*pow(x,2) -200.*x*np.exp(-100.*pow(x,2))
	elif (test=='Waveprop'):
		lamb = kwargs.get('lambda',10.0)
		u = np.sin(np.sqrt(lamb)*x)/np.sin(np.sqrt(lamb))
		du = np.sqrt(lamb)*np.cos(np.sqrt(lamb)*x)/(np.sin(np.sqrt(lamb)))
	elif (test=='Axialtruss'):
		u = -np.sin(2.3*np.pi*x)
		du = -np.cos(2.3*np.pi*x)*2.3*np.pi
	return u,du

def rhs(test,x):
#------------------------------------------
# calcula rhs de problema a resolver
#------------------------------------------
	import numpy as np
	if (test=='RW'):	
		xo = 0.4 ; alpha = 50.
		u0 = (1.0-x)*(np.arctan(alpha*(x-xo))+np.arctan(alpha*xo))
		u2 = -2.0*alpha/(1.0+(alpha*(x-xo))**2.0) \
		-(1.0-x)*(2.0*alpha**3.0*(x-xo))/(1.0+(alpha*(x-xo))**2.0)**2.0
		rhs = -u2+u0
	elif (test=='T2'):
		rhs = 6.0*x-(4.0E4*x**2.0-2.0E2)*np.exp(-1.0E2*x**2.0)
	elif (test=='Waveprop'):
		rhs = 0.0*x
	elif (test=='Axialtruss'):
		rhs = (2.3*np.pi)*(2.3*np.pi)*np.sin(2.3*np.pi*x)
	return rhs

def set_length_domain(test):
	if (test=='RW'):	
		return ( 0.0 , 1.0 )
	elif (test=='T2'):
		return ( -1.0 , 1.0 )
	elif (test=='Waveprop'):
		return ( 0.0 , 1.0 )
	elif (test=='Axialtruss'):
		return ( 0.0 , 1.0 )
	else:
		print('Error: caso no ingresado')
		return

L = set_length_domain(test)
# transformar puntos en coord. parametricas a coord. fisicas
# mapping from parametric to phisical coords
xmap = []
ctrlpts = np.linspace(L[0],L[1],n+1)
for phi_j in N:
	xmap.append( np.dot(ctrlpts,phi_j) )
xmap = np.array(xmap)

f = rhs(test,xmap)

#for i in range(len(xmap)):
#	u,du = exact_sol(test,xmap[i])
#	print(xmap[i],f[i],u,du)


### SOLUTION ###
#Initialize solution

# Find y-coordinates of control points from PDE:
def test_solver(test,N,D,D2,f,xmap):
	# NOTA
	# leer 'IGA: Neumann boundary conditions and contact', Lorenzis et al, 2014)
	if ( test == 'RW'):
		## Apply boundary conditions (Homogenous Dirichlet): 
		#D2_ = D2[1:n,1:n]
		#N_  = N[1:n,1:n]
		## o bien, 1) incluirlo en la matrix
		u_0,du_0 = exact_sol(test,xmap[0])
		u_n,du_n = exact_sol(test,xmap[-1])
		f[0] = u_0
		f[-1] = u_n
		M = -D2[:,:]+N[:,:]
		M[0,:] = N[0,:]
		M[-1,:] = N[-1,:]

		return solve(M,f)
	elif ( test == 'T2'):
		# Apply boundary conditions in rhs
	#	u_0,du_0 = exact_sol(test,xmap[0])
	#	u_n,du_n = exact_sol(test,xmap[-1])
	#	f[1] += -u_0*N[1,0]
	#	f[n-1] += -u_n*N[n-1,n]
	#	D2_ = D2[1:n,1:n]
	
		u_0,du_0 = exact_sol(test,xmap[0])
		u_n,du_n = exact_sol(test,xmap[-1])
		f[0] = u_0
		f[-1] = u_n
		M = -D2[:,:]
		M[0,:] = N[0,:]
		M[-1,:] = N[-1,:]

		return solve(M,f)
	
	elif ( test == 'Waveprop'):
		u_0,du_0 = exact_sol(test,xmap[0])
		u_n,du_n = exact_sol(test,xmap[-1])
		print(f)
		f[0] = u_0
		f[-1] = u_n
		M = D2[:,:] + 10.0*N[:,:]
		M[0,:] = N[0,:]
		M[-1,:] = N[-1,:]
		return solve(M,f)

	elif ( test == 'Axialtruss'):
		u_0,du_0 = exact_sol(test,xmap[0])
		u_n,du_n = exact_sol(test,xmap[-1])
		print(f)
		f[0] = u_0
		f[-1] = u_n
		M = D2[:,:] 
		M[0,:] = N[0,:]
		M[-1,:] = N[-1,:]
		return solve(M,f)
	
	else:
		print('Error: Caso no ingresado,')
		return

Py=test_solver(test,N,D,D2,f,xmap) 

# Create list of tuples (Px,Py) for B-spline plotting via nurbs_factory code
Px = ctrlpts

P = [(Px[i], Py[i]) for i in range(n+1)]

app_sol=[]
for phi_j in N:
	app_sol.append( np.dot(Py,phi_j) )
app_sol = np.array(app_sol)

exact,dexact = exact_sol(test,xmap)

maxerr = np.max(np.abs(exact-app_sol)) # samo y coordinatre tacaka
sampling = [t for t in xmap]

# Create a matplotlib figure
fig = plt.figure()
plt.title('B-Spline order p = %d, number of control points: %d , maxerr = %e' % (p,n+1,maxerr))
#plt.label('IgA','Analytical','Control polygon')
fig.set_figwidth(16)
ax  = fig.add_subplot(111)
# Draw the curve points
ax.scatter( xmap,app_sol, marker="o", c=sampling, cmap="jet", alpha=0.5, label = "IgA collocation" )
# Draw analytical solution
#xx=np.linspace(0.,1.,100)
ax.plot( xmap,exact, color="blue", alpha=0.7, label = "Analytical sol." )
# Draw the control cage.
#ax.plot(*zip(*P), alpha=0.3, label = "Control polygon")
plt.legend()
# Save & show figure
save_figure = True
folder='./'
if (save_figure):
	plt.savefig(folder+'RW_test1d_IGA-C_sol.svg') # utilizar formato conveniente
plt.show()



import sys
sys.exit('wea')






























### POST PROCESSING ###
# Create the Curve function
C = C_factory(P, uvec, p)

# Regularly spaced samples
xx = np.linspace(C.min, C.max, 100, endpoint=C.endpoint)
#xx = xdom

# Analytical solution
exact,dexact = exact_sol(test,xx)
#exact = np.sin(2*np.pi*xx)

# Sample the curve and make comparation with analytical solution
sampling = [t for t in xx]
curvepts = [ C(s) for s in sampling ]


# Max error
#print zip(*curvepts)[0] # samo x coordinatre tacaka
x,app = zip(*curvepts)

maxerr = np.max(np.abs(exact-app)) # samo y coordinatre tacaka
print(maxerr)

# Create a matplotlib figure
fig = plt.figure()
plt.title('B-Spline order p = %d, number of control points: %d , maxerr = %e' % (p,n+1,maxerr))
#plt.label('IgA','Analytical','Control polygon')
fig.set_figwidth(16)
ax  = fig.add_subplot(111)
# Draw the curve points
ax.scatter( *zip(*curvepts), marker="o", c=sampling, cmap="jet", alpha=0.5, label = "IgA collocation" )
# Draw analytical solution
#xx=np.linspace(0.,1.,100)
ax.plot( xx,exact, color="blue", alpha=0.7, label = "Analytical sol." )
# Draw the control cage.
ax.plot(*zip(*P), alpha=0.3, label = "Control polygon")
plt.legend()
# Save & show figure
save_figure = True
folder='./'
if (save_figure):
	plt.savefig(folder+'RW_test1d_IGA-C_sol.svg') # utilizar formato conveniente
plt.show()
