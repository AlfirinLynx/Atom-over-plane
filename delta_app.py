import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy import interpolate


from sympy.physics.wigner import wigner_3j
from scipy.special import eval_legendre, factorial, factorial2

#Условие - лямбда на стенке
lam = - 0.1

#Шаг по координате
s = 0.0001

#Расстояние от стенки
h = 0.3

#Конечная точка
rf = 10

#Параметры дельта-потенциала
d = 0.011
u  = lam/(2*d)

#Положение ядра
a =  h + d


#Отступ в нуле
r0 = 0.001
#Первый интервал, 0т 0 до d
rar1 = np.linspace(r0, d, int((d-r0)/s), dtype='float64')

#Второй интервал, от d до a
rar2 = np.linspace(d, a, int(h/s), dtype='float64')

#Третий интервал, от a до inf
rar3 = np.linspace(a, rf, int((rf-a)/s), dtype='float64')

#Массив от d до inf
rar23 = np.append(rar2[:-1], rar3)

#Массив d/r
drar = d/rar23
drari = np.fliplr([drar])[0] #инвертированный



# up to 2*N harmonics(all even), ind[l] = l - (n-k)
N = 5  #число гармоник 2*N

#Табулируем 3_j символы
W = np.array([[[np.power(np.float64(wigner_3j(2*n, 2*k, 2*l, 0, 0, 0).n(32)), 2) for l in range(n-k, n+k+1)] for k in range(n+1)] for n in range(N+1)])


#print(Nw)

F = [] #Interpolation functions for Legendre
# Legendre l = 2*n +1, n = (l-1)//2
L = np.zeros((2*N+1, np.size(drari)))
for n in range(2*N+1):
	L[n] = eval_legendre(2*n+1, drari)
	F.append(interpolate.interp1d(drari, L[n]))



"""
rnew = np.linspace(d/rf, 1, 200000)
y  = F[1](rnew)


plt.plot( drari, L[1], 'o', rnew, y, '-')
plt.show()
"""

def ak(k):
	if k == 0: return 1
	return factorial2(2*k-1, exact=True)/factorial(k, exact=True)


#Заполняем массив коэф-тов ak max: n+m = 2N + 2N = 4N
Ak = np.array([ak(k) for k in range(4*N+1)], dtype='float64')

# Интеграл от произведения полиномов Лежандра
def pnpm(i, j, r):
	if i>j: 
		n, m = 2*i, 2*j
	else:
		n, m = 2*j, 2*i

	def pint(l, r):
		if l == 0: return 2*r
		#return 2*(F[l//2](r) - F[(l-2)//2](r))/(2*l+1)
		return 2*(eval_legendre(l+1, r) - eval_legendre(l-1, r))/(2*l+1)

	A = np.array([ pint(n+m-2*k, r) for k in range(m+1)], dtype='float64')
	B = np.array([(2*(n+m) - 4*k + 1) * Ak[m-k]*Ak[k]*Ak[n-k]/(Ak[n+m-k] * (2*(n+m) - 2*k + 1)) for k in range(m+1)], dtype='float64')
	return np.inner(A, B)

#Вспомогательные функции, 1 - для r < a, 2 - для r > a

def wl1(l, r):
	return np.power(r/a, l)/a

def wl2(l, r):
	return np.power(a/r, l)/r

def b1(n, k, r):
	if n>k:
		mx, mn = n, k
	else:
		mx, mn = k, n
	wa = np.array([wl1(2*l,r) for l in range(mx - mn, mx + mn + 1)])
	return (4*k+1)*np.inner(wa, W[mx][mn])


def b2(n, k, r):
	if n>k:
		mx, mn = n, k
	else:
		mx, mn = k, n
	wa = np.array([wl2(2*l,r) for l in range(mx - mn, mx + mn + 1)])
	return (4*k+1)*np.inner(wa, W[mx][mn])
	



print(b1(5, 5, 0.2))
