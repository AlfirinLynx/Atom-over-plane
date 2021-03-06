import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint
from scipy import interpolate


from sympy.physics.wigner import wigner_3j
from sympy import integrate as sint
from sympy import Symbol, legendre
from scipy.special import eval_legendre, factorial, factorial2

import mpmath as mp
mp.mp.dps = 30
mp.mp.pretty = True



met5 = 'dopri5'
met8 = 'dop853'



#Условие - лямбда на стенке
lam = 0.8

#Шаг по координате
s1 = 0.0001
s2 = 0.01
s3 = 0.01

#Расстояние от стенки
h = 2.37


#Параметры дельта-потенциала
d = 0.001
u  = lam/(2*d)

#Положение ядра
a =  h + d

#Конечная точка
rf = 10.0 * a


#Отступ в нуле
r0 = 0.0001
#Первый интервал, 0т 0 до d
rar1 = np.linspace(r0, d, int((d-r0)/s1), dtype='float64')

#Второй интервал, от d до a
rar2 = np.linspace(d, a, int(h/s2), dtype='float64')

#Третий интервал, от a до inf
rar3 = np.linspace(a, rf, int((rf-a)/s3), dtype='float64')

#Массив от d до inf
rar23 = np.append(rar2[:-1], rar3)

#Массив d/r
drar = d/rar23
drari = np.fliplr([drar])[0] #инвертированный



# up to 2*N harmonics(all even), ind[l] = l - (n-k)
N = 2  #число гармоник 2*N

#Табулируем 3_j символы
W = np.array([[[np.float64((wigner_3j(2*n, 2*k, 2*l, 0, 0, 0) ** 2).n(70)) for l in range(n-k, n+k+1)] for k in range(n+1)] for n in range(N+1)])

#Табулируем комбинацию из трех полиномов Лежандра
#Параметр обрезания бесконечной суммы
Cl = 160

def f(n1, n2, n3):
    def p(x):
        return legendre(n1, x) * legendre(n2, x) * legendre(n3, x)

    return p

x = Symbol('x', dummy = True)

L = np.array([[[ np.float64(sint(f(2*n, 2*k, 2*l+1)(x), (x, 0, 1)).n(70))  for l in range(Cl+1)] for k in range(n+1)] for n in range(N+1)])


    

print(L)

#print(Nw)
"""
F = [] #Interpolation functions for Legendre
# Legendre l = 2*n +1, n = (l-1)//2
L = np.zeros((2*N+1, np.size(drari)))
for n in range(2*N+1):
	L[n] = eval_legendre(2*n+1, drari)
	F.append(interpolate.interp1d(drari, L[n]))
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


def k1(n, k, r):
    if n>k:
        mx, mn = n, k
    else:
        mx, mn = k, n
    wa = np.array([wl1(2*l+1,r) for l in range(Cl + 1)])
    return (4*k+1)*np.inner(wa, L[mx][mn])

def k2(n, k, r):
    if n>k:
        mx, mn = n, k
    else:
        mx, mn = k, n
    wa = np.array([wl2(2*l+1,r) for l in range(Cl + 1)])
    return (4*k+1)*np.inner(wa, L[mx][mn])



########Численное интегрирование

###Правая часть

# На вход - y = [u0, u2, ..., u2N, u0', u2', ..., u2N']
# 0 < r < d
def coulomb1(r, y, e, u, d):
    right_exp = np.zeros(2*N+2) #Правая часть - 0, 2, ..., 2N + столько же производных
    #right_exp = [0 for i in range(2*N+2)]
        
    for k in range(N+1):
        right_exp[k] = y[N+1+k]
        uv = np.array(y[:N+1]) #Только ф-ции u
        b = np.array([b1(n, k, r) for n in range(N+1)])
        t = np.array([k1(n, k, r) for n in range(N+1)])
        right_exp[N+1+k] = (2*k*(2*k+1)/r**2 - 2*e)*y[k] - 2*np.inner(uv, b) - 2*np.inner(uv, t) + 2*u*y[k]
    return right_exp

# d < r < a
def coulomb2(r, y, e, u, d):
    right_exp = np.zeros(2*N+2) #Правая часть - 0, 2, ..., 2N + столько же производных
    for k in range(N+1):
        right_exp[k] = y[N+1+k]
        uv = np.array(y[:N+1]) #Только ф-ции u
        b = np.array([b1(n, k, r) for n in range(N+1)])
        t = np.array([k1(n, k, r) for n in range(N+1)])
        p = np.array([pnpm(k, n, d/r) for n in range(N+1)])
        right_exp[N+1+k] = (2*k*(2*k+1)/r**2 - 2*e)*y[k] - 2*np.inner(uv, b) - 2*np.inner(uv, t) + (4*k+1)*u*np.inner(uv, p)
    return right_exp
        
# a < r < inf
def coulomb3(r, y, e, u, d):
    right_exp = np.zeros(2*N+2) #Правая часть - 0, 2, ..., 2N + столько же производных
    for k in range(N+1):
        right_exp[k] = y[N+1+k]
        uv = np.array(y[:N+1]) #Только ф-ции u
        b = np.array([b2(n, k, r) for n in range(N+1)])
        t = np.array([k2(n, k, r) for n in range(N+1)])
        p = np.array([pnpm(k, n, d/r) for n in range(N+1)])
        right_exp[N+1+k] = (2*k*(2*k+1)/r**2 - 2*e)*y[k] - 2*np.inner(uv, b) - 2*np.inner(uv, t) + (4*k+1)*u*np.inner(uv, p)
    return right_exp


###Начальные условия в нуле, строим N+1 линейно независимых решений
Nl = N

def shootR(e, u, d):
    print("Solving...")
    for k in range(Nl+1):
        #Начальные условия на первом интервале
        y0 = np.zeros(2*N+2)
        y0[k] = np.power(r0, 2*k+1)
        y0[N+1+k] = (2*k+1)*np.power(r0, 2*k)
                
        sol1 = np.zeros((np.size(rar1), 2*N+2))
        sol1[0] = y0
                
        s = ode(coulomb1).set_integrator('dopri5')
        s.set_initial_value(y0, r0).set_f_params(e, u, d)

        i = 1
        for r in rar1[1:]:
            sol1[i] = s.integrate(r)
            if not  s.successful(): print("error!")
            print("i = %s, y = %s" % (i, sol1[i][k]))
            i += 1
                
             

        print("Plotting")
        plt.plot(rar1, sol1[:, k], 'b', label='Solution at 0 < r < d, k=%s' % k)
        plt.legend(loc='best')
        plt.xlabel('r')
        plt.grid()
        plt.show()

        #Перенормировка
        norm =np.amax(np.absolute(sol1[-1]))
        y0  = sol1[-1]/norm

        #Начальные условия на втором интервале
        sol2 = np.zeros((np.size(rar2), 2*N+2))
        sol2[0] = y0

        s = ode(coulomb2).set_integrator('dopri5')
        s.set_initial_value(y0, d).set_f_params(e, u, d)

        i = 1
        mid = np.size(rar2)//2
        for r in rar2[1:]:
            sol2[i] = s.integrate(r)
            if not  s.successful(): print("error!")
            print("i = %s, y = %s" % (i, sol2[i][k]))
            if i == mid: break
            i += 1

        
        #Перенормировка
        norm =np.amax(np.absolute(sol2[mid]))
        y0  = sol2[mid]/norm

        s = ode(coulomb2).set_integrator('dopri5')
        s.set_initial_value(y0, rar2[mid]).set_f_params(e, u, d)

        i = mid+1
        for r in rar2[i:]:
            sol2[i] = s.integrate(r)
            if not  s.successful(): print("error!")
            print("i = %s, y = %s" % (i, sol2[i][k]))
            i += 1
        

        print("Plotting")
        plt.plot(rar2[mid+1:], sol2[mid+1:, k], 'b', label='Solution at d < r < a, k=%s' % k)
        plt.legend(loc='best')
        plt.xlabel('r')
        plt.grid()
        plt.show()

def shootL(e, u, d):
    print("Solving...")
    for k in range(Nl+1):
        #Начальные условия на бесконечности
        gam = np.sqrt(-2*e)
        y0 = np.zeros(2*N+2)
        y0[k] = np.exp(-gam*rf)
        y0[N+1+k] =-gam * np.exp(-gam*rf)

        l = np.size(rar3)
        sol3 = np.zeros((l, 2*N+2))
        sol3[l-1] = y0
        
        s = ode(coulomb3).set_integrator('dopri5')
        s.set_initial_value(y0, rf).set_f_params(e, u, d)

        mid = 14*l//15
        print('mid = %s' % mid)
        for i in range(1, mid+1):
            sol3[l-1-i] = s.integrate(rar3[l-1-i])
            if not  s.successful(): print("error!")
            print("i = %s, y = %s" % (i, sol3[l-1-i][k]))

        #Перенормировка
        norm =np.amax(np.absolute(sol3[l-1-mid]))
        y0  = sol3[l-1-mid]/norm

        s = ode(coulomb3).set_integrator('dopri5')
        s.set_initial_value(y0, rar3[l-1-mid]).set_f_params(e, u, d)

        for i in range(mid+1, l):
            sol3[l-1-i] = s.integrate(rar3[l-1-i])
            if not  s.successful(): print("error!")
            print("i = %s, y = %s" % (i, sol3[l-1-i][k]))

        print("Plotting")
        plt.plot(rar3[:l-1-mid], sol3[:l-1-mid, k], 'b', label='Solution at a < r < inf, k=%s' % k)
        plt.legend(loc='best')
        plt.xlabel('r')
        plt.grid()
        plt.show()
        
def shootRq(e, u, d, dar, meth = met5):
    print("Solving...")
    mid = (a+d)/2.0
    for k in range(Nl+1):
        #Начальные условия на первом интервале
        y0 = np.zeros(2*N+2)
        y0[k] = np.power(r0, 2*k+1)
        y0[N+1+k] = (2*k+1)*np.power(r0, 2*k)

        s = ode(coulomb1).set_integrator(meth)
        s.set_initial_value(y0, r0).set_f_params(e, u, d)

        y0 = s.integrate(d)

        print('At r = d, sol for k = %s' % k)
        print(y0[k])
        
        
        print('renorming...')
        norm =np.amax(np.absolute(y0))
        y0  = y0/norm

        print("Solving...")
        s = ode(coulomb2).set_integrator(meth)
        s.set_initial_value(y0, d).set_f_params(e, u, d)
        y0 = s.integrate(mid)

        print('renorming...')
        norm =np.amax(np.absolute(y0))
        y0  = y0/norm

        print('Solving...')
        s = ode(coulomb2).set_integrator(meth)
        s.set_initial_value(y0, mid).set_f_params(e, u, d)
        y0 = s.integrate(a)
        dar[k] = y0

        print('At r = a, sol for k = %s, e = %s' % (k, e))
        print(y0[k])
        
        
        
    




    
def shootLq(e, u, d, dar, meth = met5):
    print("Solving...")
    mid = (a + rf)/2.0
    rep = a + (rf-a)/15.0
    for k in range(Nl+1):
        #Начальные условия на бесконечности
        gam = np.sqrt(-2*e)
        y0 = np.zeros(2*N+2)
        y0[k] = np.exp(-gam*rf)
        y0[N+1+k] =-gam * np.exp(-gam*rf)

        s = ode(coulomb3).set_integrator(meth)
        s.set_initial_value(y0, rf).set_f_params(e, u, d)

        y0 = s.integrate(mid)
        print('renorming at r = %s' % mid)
        norm =np.amax(np.absolute(y0))
        y0  = y0/norm

        print('Solving...')
        s = ode(coulomb3).set_integrator(meth)
        s.set_initial_value(y0, mid).set_f_params(e, u, d)
        y0 = s.integrate(rep)
        print('renorming at r = %s' % rep)
        norm =np.amax(np.absolute(y0))
        y0  = y0/norm

        s = ode(coulomb3).set_integrator(meth)
        s.set_initial_value(y0, rep).set_f_params(e, u, d)
        y0 = s.integrate(a)


        dar[N+k+1] = y0

        print('At r = a, sol for k = %s, e = %s' % (k, e))
        print(y0[k])


def wronsk(e0,e1, st, u, d):
    e = e0
    detar = []
    enar = []
    while e <= e1:
        dar  = np.zeros((2*N+2, 2*N+2))
        shootRq(e, u, d, dar)
        shootLq(e, u, d, dar)
        M = mp.matrix(2*N+2, 2*N+2)
        for i in range(2*N+2):
            for j in range(2*N+2):
                M[i,j] = mp.mpf(str(dar[i,j]))
        det = mp.det(M)
        #det = np.linalg.det(dar)
        print("det=%s" % det)
        print(dar)
        detar.append(det)
        enar.append(e)
        e += st

    print("Plotting")
    plt.plot(enar, detar, 'b', label='Wronskian d=%s, h = %s, u = %s, lambda = %s' % (d, h, u, lam))
    plt.legend(loc='best')
    plt.xlabel('e')
    plt.grid()
    plt.show()

    
        
                      

        
            

        
        


e = -1.5
wronsk(-0.53, -0.48, 0.005, u, d)



