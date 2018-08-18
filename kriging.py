from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

import itertools
import matplotlib
import numpy as np

def f(x, y):
    return -1* pylab.cos(x) + 2 * pylab.sin(y)
    #return x*x + y*y 

xachs = np.arange(0, 2, 0.01)
yachs = np.arange(0, 2, 0.01)
dataLen = len(xachs)

stDev = np.sqrt(0.01)
#X = np.random.normal(0,stDev,(dataLen, dataLen))
z = np.random.normal(0,stDev,size=[209,200])
bm_1 = np.append([[0]*200], np.cumsum(z, axis=0), axis=0)
bm_2 = np.append([[0]*200], np.cumsum(z, axis=1), axis=0)
bm = bm_1 + bm_2
X = bm[10:,:] - bm[0:200,:]

m = np.zeros([len(xachs), len(yachs)])
datapoints = list(itertools.product(np.arange(0, 2, 0.5), np.arange(0, 2, 0.5)))
#print(datapoints)




for i in xrange(len(xachs)):
    for j in xrange(len(yachs)):
        m[i, j] = f(xachs[i], yachs[j])

Y = m+X
realisation = Y[::50,::50].ravel()
print(np.shape(realisation))

plt.pcolor(xachs, yachs, Y)

plt.show()

def kriging(datapoints, covariancefunction, realisation):

    Sigma = np.zeros([len(datapoints), len(datapoints)])

    for i, data1 in enumerate(datapoints):
        for j, data2 in enumerate(datapoints):
            Sigma[i,j] = covariancefunction(data1, data2)
   

    left = np.append(Sigma, np.ones((len(datapoints),1)), 1) 
    #print(left)
    a = np.append(np.ones(len(datapoints)), [[0]])
    #print(a)
    left = np.append(left, [a], 0) + np.eye(len(datapoints)+1) * 0.001
    #print(left)

    estY = np.zeros([len(np.arange(0, 2, 0.1)), len(np.arange(0, 2, 0.1))])
    for k, t_1 in enumerate(np.arange(0, 2, 0.1)):
        for l, t_2 in enumerate(np.arange(0, 2, 0.1)):
            t = np.array([t_1, t_2])

            Sigma_t = np.zeros(len(datapoints))
            for i, data1 in enumerate(datapoints):
                Sigma_t[i] = covariancefunction(np.array(data1), t)
            
            right = np.append(Sigma_t, [1])
            #print(t)
            #print(right)
            #print(np.linalg.solve(left, right))
            lsg = np.linalg.solve(left, right)
            lamb = lsg[0:-1]
            estY[k,l] = np.dot(lamb, np.transpose(realisation))
            print(right)
    #print(realisation)

    return estY



def covariancefunction(x,y):
    t = x[0]+x[1]
    s = y[0]+y[1]
    if abs(t-s) < 0.001:
        return 2 * (t<=s) * t + 2 * (t>s) * s  - (t<=s+0.10) * t - (s+0.10<t) * (s+0.10) - (t+0.10<=s) * (t+0.10) - (s<t+0.10) * s + 0.10
    else:
        return 0
    


estY = kriging(datapoints, covariancefunction, realisation)

plt.pcolor(np.arange(0, 2, 0.1), np.arange(0, 2, 0.1), estY)

plt.show()