import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint
from scipy import interpolate

from scipy.special import eval_legendre, factorial, factorial2

import mpmath as mp
mp.mp.dps = 300
mp.mp.pretty = True



met5 = 'dopri5'
met8 = 'dop853'

#Гармоники и относ. ошибка
N = 2
NP = 2


#Условие - лямбда на стенке
lam = -0.3

#Расстояние от стенки
h = 0.308


#Параметры дельта-потенциала
d = 0.001
u  = lam/(2*d)

#Положение ядра
a =  h + d

#Конечная точка
rf = 10.0 * a


#Отступ в нуле
r0 = 0.0001


#Чтение файлов

def ReadF(name, ar):
    with open(name, 'r') as f:
        for n in range(N+1):
            for k in range(n+1):
                words = f.readline().split()
                for l in range(len(words)//2):
                   # p = np.float64(words[2*l])
                    #q = np.float64(words[2*l+1])
                    p = mp.mpf(words[2*l])
                    q = mp.mpf(words[2*l+1])
                    #print(p, q, "\t", end = "")
                    ar[n][k][l] = np.float64(p/q)
               # print("")

NumWigFile = "NumWig%sx%spow%sCppOut.dat" % (N, N, NP)

HalfWigFile = "HalfWig%sx%spow%sCppOut.dat" % (N, N, NP)

WigFile = "wig%s.dat" % N

NumW = np.array([ [np.int(0) for k in range(n+1)] for n in range(N+1)])

cl = 0
for line in open(NumWigFile):
    words = line.split()
    for k in range(len(words)):
        NumW[cl][k] = np.int(words[k])
    cl += 1
        
        

W = np.array([[[0.0 for l in range(n-k, n+k+1)] for k in range(n+1)] for n in range(N+1)])

L = np.array([[[0.0  for l in range(NumW[n][k] + 1)] for k in range(n+1)] for n in range(N+1)])

ReadF(WigFile, W)

ReadF(HalfWigFile, L)



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
    ct = NumW[mx][mn]
    wa = np.array([wl1(2*l+1,r) for l in range(ct + 1)])
    return (4*k+1)*np.inner(wa, L[mx][mn])

def k2(n, k, r):
    if n>k:
        mx, mn = n, k
    else:
        mx, mn = k, n
    ct = NumW[mx][mn]
    wa = np.array([wl2(2*l+1,r) for l in range(ct + 1)])
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

    
        
wronsk(-0.97, -0.8, 0.01, u, d)
