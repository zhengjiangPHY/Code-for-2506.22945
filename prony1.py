# QNM frequency evaluation by Prony method
# latest update 20200308.1519

#   """
#     Input  : real arrays t, F of the same size (ti, Fi)
#            : integer m - the number of modes in the exponential fit
#     Output : arrays a and b such that F(t) ~ sum ai exp(bi*t)
#     Usage  : If you have arrays t and F, it can be called as:
#              a_est, b_est = prony(t, F, m)
#     Comment: the formulae referred to below are from arXiv:1102.4014
#
#              Amat (N-m)x(m) matrix X, see the Eq. below (3.47)
#              bmat (N-m)x(1) column matrix, see the r.h.s. of the Eq
#
#              np.linalg.lstsq(Amat, bmat)
#              return the least-squares solution to a linear matrix equation
#                Amat d = bmat 
#              where the equation is usually over-determined in practice
#              it is invoked in step 1 to find the coefficient d
#              a derivation of the resultant algorithm can be found here
#              https://s-mat-pcs.oulu.fi/~mpa/matreng/ematr5_5.htm
#              
#              c (m+1) polinomial coefficients of (3.46) obtained from d
#              one notes that the negative sign is from Eq below (3.47)
#              
#              poly.polyroots(c)
#              compute the roots of a polynomial whose coefficients are c
#              
#              b_est are determined by the definition of z_j from (3.45)
#
#              np.linalg.lstsq(Amat, bmat)
#              is invoked in step 3 to solve the matrix Eq. (3.45)
#
#     http://sachinashanbhag.blogspot.com/2017/09/prony-method.html
#     Posted by Sachin Shanbhag 
#   """

import os, sys
import numpy as np
import numpy.polynomial.polynomial as poly
import re
import math

def prony(t, F, m):
    # Solve LLS problem in step 1
    # Amat is (N-m)*m and bmat is N-m*1
    N    = len(t)
    Amat = np.zeros((N-m, m),dtype=complex)
    bmat = F[m:N]

    for jcol in range(m):
        Amat[:, jcol] = F[m-jcol-1:N-1-jcol]
        
    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    d = sol[0]

    # Solve the roots of the polynomial in step 2
    # first, form the polynomial coefficients
    c = np.zeros((m+1),dtype=complex)
    c[m] = 1.
    for i in range(1,m+1):
        c[m-i] = -d[i-1]

    u = poly.polyroots(c)
    b_est = np.log(u)/(t[1] - t[0])

    # Set up LLS problem to find the "a"s in step 3
    Amat = np.zeros((N, m),dtype=complex)
    bmat = F

    for irow in range(N):
        Amat[irow, :] = u**irow
        
    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    a_est = sol[0]

    return a_est, b_est

def readFt(fileName='outputFt_mat.txt'):
    fraction_pattern = re.compile(r"^(?P<num>[0-9]+)/(?P<den>[0-9]+)$")
    if not os.path.isfile(fileName):
        print('...fatal error! The file ',fileName,' does not exist!')
        exit(1)
    t = []
    F = []
    data = open( fileName, 'r' ).readlines()
    print('...reading file:', fileName)
    isFirstLine = True
    indexp = 0
    for line in data:
        linesplit = " ".join(line.split()).split(' ')
        if isFirstLine:
            isFirstLine = False
            print('...the content of the first line: ',linesplit)
            numberOfItems =  len(linesplit)
            if numberOfItems != 4:
                print('Fatal error: the number of items does not match the standard. program halt!')
                print('numberOfItems:',numberOfItems)
                print('troublesome content:',linesplit)
                exit(1)
        indexp += 1
        g = fraction_pattern.search(linesplit[0])
        if g:
            itemTinput = float(g.group("num"))/float(g.group("den"))
        else:
            itemTinput = float(linesplit[0])

        t += [ complex(itemTinput, float(linesplit[1])) ]
        F += [ complex(float(linesplit[2]), float(linesplit[3])) ]
    return t, F

def createFt(fileName='outputFt.txt'):
    t=np.linspace(0., 10.0, num=50)
    F=1.0*np.exp(-(2.+2.j)*t)+0.2*np.exp((-6.+5.j)*t)
    f = open( fileName, 'w' ) #clean up existing content
    f.close
    for tValue, FValue in zip(t, F):
        f = open( fileName, 'a' )
        f.write("%7.5f"%tValue.real + ' ' + "%7.5f"%tValue.imag + ' ' + "%25.20f"%FValue.real + ' ' + "%25.20f"%FValue.imag + '\n' )
        f.close()

if __name__ == '__main__':
  
    print(sys.argv)
    print("hello world!")
    print(sys.float_info)
#    fileName = 'outputFt.txt'
#    createFt( fileName )
    fileName = 'outputFt_matg_m1q9nc_l1.txt'
    t, F = readFt( fileName )
#    print(t,F)
    m=10
    a_est, b_est = prony(t, F, m)
    print('a (weight):\n',a_est)
    print('b (omega):\n',b_est)
#    print('the expected dominate frequency for toy model is: ',math.sqrt(pow(math.pi,2)-1.0))
    print('program halt! goodbye and happy debudding!')
    exit(0)
