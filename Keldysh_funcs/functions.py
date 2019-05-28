#
import numpy as np
import math
#
from scipy.special import ellipk, ellipe, dawsn
#scipy.special.ellipk(m): Complete elliptic integral of the first kind
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html#scipy.special.ellipk
#scipy.special.ellipe(m): Complete elliptic integral of the second kind
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html#scipy.special.ellipe
#scipy.special.dawsn(x): Dawson's integral.
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.dawsn.html#scipy.special.dawsn
#Mathematical constants
pi = np.pi
tpi = 2.0*pi
fpi = 4.0*pi
#functions
def Q_old(gamma,x):
    gamma1 = gamma**2/(1.0 + gamma**2)
    gamma2 = 1.0 - gamma1
    N = 1000
    Q = 0.0
    for n in range(N):
        arg1 = 0.5*pi**2*(math.floor(x + 1) - x + float(n))/(ellipk(gamma2)*ellipe(gamma2))
        arg2 = np.sqrt(arg1)
        Q = Q + np.exp(-float(n)*pi*(ellipk(gamma1) - ellipe(gamma1))/ellipe(gamma2))*dawsn(arg2)
    Q = Q*np.sqrt(0.5*pi/ellipk(gamma2))
    return Q
#
def Q(gamma,x):
    gamma1 = gamma**2/(1.0 + gamma**2)
    gamma2 = 1.0 - gamma1
    N = 10000
    epsconv = 1.0e-8
    Q = 0.0
    for n in range(N):
        arg1 = 0.5*pi**2*(math.floor(x + 1) - x + float(n))/(ellipk(gamma2)*ellipe(gamma2))
        arg2 = np.sqrt(arg1)
        dQ = np.exp(-float(n)*pi*(ellipk(gamma1) - ellipe(gamma1))/ellipe(gamma2))*dawsn(arg2)
        Q = Q + dQ
        if (dQ/Q < epsconv):
            break
        if (n == N-1):
            char = 'WARNING!: Maximum integer, '+str(N)+', in Q function is not enough.'
            print(char)
    Q = Q*np.sqrt(0.5*pi/ellipk(gamma2))
    return Q
#
#
def get_gamma_x(m,delta,omega,F):
    gamma = omega*np.sqrt(m*delta)/F
    gamma1 = gamma**2/(1.0 + gamma**2)
    gamma2 = 1.0 - gamma1
    x = 2.0*delta*ellipe(gamma2)/(pi*omega*np.sqrt(gamma1))
    return gamma, gamma1, gamma2, x
#
def get_W_old(m,delta,omega,F):
    gamma, gamma1, gamma2, x = get_gamma_x(m,delta,omega,F)
    W = 2.0*2.0*omega/(9.0*pi)*(m*omega/np.sqrt(gamma1))**1.5*\
    Q_old(gamma,x)*np.exp(-pi*math.floor(x + 1)*(ellipk(gamma1) - ellipe(gamma1))/ellipe(gamma2))
    return W
#
def get_W(m,delta,omega,F):
    gamma, gamma1, gamma2, x = get_gamma_x(m,delta,omega,F)
    W = 2.0*2.0*omega/(9.0*pi)*(m*omega/np.sqrt(gamma1))**1.5*\
    Q(gamma,x)*np.exp(-pi*math.floor(x + 1)*(ellipk(gamma1) - ellipe(gamma1))/ellipe(gamma2))
    return W
#
def get_WMP(m,delta,omega,F):
    gamma, gamma1, gamma2, x = get_gamma_x(m,delta,omega,F)
    power = math.floor(x + 1.0)
    WMP = 2.0*2.0*omega/(9.0*pi)*(m*omega)**1.5*\
    dawsn(np.sqrt(2.0*power - 2.0*x))*np.exp(2.0*power*(1.0 - 0.25/gamma**2))*(0.0625/gamma**2)**power
    return WMP
#
def get_WTUN(m,delta,omega,F):
    gamma, gamma1, gamma2, x = get_gamma_x(m,delta,omega,F)
#In order to avoid overflow in exponential argument following line, threshold is introduced, around which actual value is not trustable.
    arg = min(-0.5*pi*delta*gamma/omega*(1.0 - 0.125*gamma**2),30.0*np.log(10.0))
    WTUN = 2.0*2.0/(9.0*pi**2)*delta*(m*delta)**1.5*(omega/(delta*gamma))**2.5*np.exp(arg)
    return WTUN
#
def get_WTUNYS(m,delta,omega,F):
    gamma, gamma1, gamma2, x = get_gamma_x(m,delta,omega,F)
    arg = -0.5*pi*delta*gamma/omega
    WTUNYS = 2.0*2.0/(9.0*pi**2)*delta*(m*delta)**1.5*(omega/(delta*gamma))**2.5*np.exp(arg)
    return WTUNYS
