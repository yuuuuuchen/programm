from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import itertools
import matplotlib
import numpy as np

def f(x):
    #return 3
    return 1* pylab.cos(x) + 2 * pylab.sin(2*x)
    #return 0.01* pylab.cos(x) + 0.02 * pylab.sin(x)

def covariancefunction(t,s,h):
    #return (t<=s) * t + (s<t) * s
    return 2 * (t<=s) * t + 2 * (t>s) * s  - (t<=s+h) * t - (s+h<t) * (s+h) - (t+h<=s) * (t+h) - (s<t+h) * s + h

def genCov(d, h): 
    #return 0.5*(d>0)*(-d) + 0.5*(d<=0)*d
    if d>=0:
        return h - (h<=d) * h - (d<h) * d
    elif d<0:
        return h + d - (h+d) * (h+d<0)

def drift_1(x):
    return pylab.sin(2*x)

def drift_2(x):
    return pylab.cos(x)


def kriging(order, datapoints, realisation, covariancefunction=None, drift=None, genCov=None):
    # Kovarianzfunktion wird angegeben
    if covariancefunction != None:    
        # ordinary Kriging
        if isinstance(order, int) and order == -1: 
            Sigma = np.zeros([len(datapoints), len(datapoints)])
            for i, data1 in enumerate(datapoints):
                for j, data2 in enumerate(datapoints):
                    Sigma[i,j] = covariancefunction(data1, data2, 0.15)
            left = np.append(Sigma, np.ones((len(datapoints),1)), 1) 
            a = np.append(np.ones(len(datapoints)), [[0]])
            left = np.append(left, [a], 0) #+ np.eye(len(datapoints)+1) * 0.001
    
            estY = np.zeros(len(np.arange(0, 2, 0.01)))
            for k, t in enumerate(np.arange(0, 2, 0.01)):
                Sigma_t = np.zeros(len(datapoints))
                for l, data in enumerate(datapoints):
                    Sigma_t[l] = covariancefunction(data, t, 0.15)
                right = np.append(Sigma_t, [1])

                lsg = np.linalg.solve(left, right)
                lamb = lsg[0:-1]
                estY[k] = np.dot(lamb, np.transpose(realisation))
        
        # intrinsic Kriging
        elif isinstance(order, int) and order >= 0:
            left = np.zeros([len(datapoints) + len(drift) + order + 1, len(datapoints) + len(drift) + order + 1])
            for i, data1 in enumerate(datapoints):
                for j, data2 in enumerate(datapoints):
                    left[i,j] = covariancefunction(data1, data2, 0.15)

            for m, data in enumerate(datapoints):
                for n, d in enumerate(drift):
                    left[m, len(datapoints) + n] = d(data)
                    left[len(datapoints) + n, m] = d(data)
            
            for m, data in enumerate(datapoints):
                for n in range(order+1):
                    left[m, len(datapoints) + len(drift) + n] = data**n
                    left[len(datapoints) + len(drift) + n, m] = data**n 

            estY = np.zeros(len(np.arange(0, 2, 0.01)))
            for k, t in enumerate(np.arange(0, 2, 0.01)):
                right = np.zeros(len(datapoints) + len(drift) + order + 1)
                for o, data in enumerate(datapoints):
                    right[o] = covariancefunction(data, t, 0.15)
                for p, d in enumerate(drift):
                    right[len(datapoints) + p] = d(t)
                for q in range(order+1):
                    right[len(datapoints) + len(drift) + q] = t**q
                
                lsg = np.linalg.solve(left, right)
                leftout = len(drift) + order + 1
                lamb = lsg[0:-leftout]
                estY[k] = np.dot(lamb, np.transpose(realisation))
        else:
            raise TypeError("Die Variable 'order' muss eine Natuerliche Zahl oder -1 sein")

        return estY
    # generalisierten Kovarianzfunktion wird angegeben
    elif genCov != None:
        # ordinary Kriging
        if isinstance(order, int) and order == -1: 
            Sigma = np.zeros([len(datapoints), len(datapoints)])
            for i, data1 in enumerate(datapoints):
                for j, data2 in enumerate(datapoints):
                    Sigma[i,j] = genCov(data2-data1, 0.15)
            left = np.append(Sigma, np.ones((len(datapoints),1)), 1) 
            a = np.append(np.ones(len(datapoints)), [[0]])
            left = np.append(left, [a], 0) #+ np.eye(len(datapoints)+1) * 0.001
    
            estY = np.zeros(len(np.arange(0, 2, 0.01)))
            for k, t in enumerate(np.arange(0, 2, 0.01)):
                Sigma_t = np.zeros(len(datapoints))
                for l, data in enumerate(datapoints):
                    Sigma_t[l] = genCov(t-data, 0.15)
                right = np.append(Sigma_t, [1])

                lsg = np.linalg.solve(left, right)
                lamb = lsg[0:-1]
                estY[k] = np.dot(lamb, np.transpose(realisation))
        
        # intrinsic Kriging
        elif isinstance(order, int) and order >= 0:
            left = np.zeros([len(datapoints) + len(drift) + order + 1, len(datapoints) + len(drift) + order + 1])
            for i, data1 in enumerate(datapoints):
                for j, data2 in enumerate(datapoints):
                    left[i,j] = genCov(data2-data1, 0.15)

            for m, data in enumerate(datapoints):
                for n, d in enumerate(drift):
                    left[m, len(datapoints) + n] = d(data)
                    left[len(datapoints) + n, m] = d(data)
            
            for m, data in enumerate(datapoints):
                for n in range(order+1):
                    left[m, len(datapoints) + len(drift) + n] = data**n
                    left[len(datapoints) + len(drift) + n, m] = data**n 

            estY = np.zeros(len(np.arange(0, 2, 0.01)))
            for k, t in enumerate(np.arange(0, 2, 0.01)):
                right = np.zeros(len(datapoints) + len(drift) + order + 1)
                for o, data in enumerate(datapoints):
                    right[o] = genCov(t-data, 0.15)
                for p, d in enumerate(drift):
                    right[len(datapoints) + p] = d(t)
                for q in range(order+1):
                    right[len(datapoints) + len(drift) + q] = t**q
                
                lsg = np.linalg.solve(left, right)
                leftout = len(drift) + order + 1
                lamb = lsg[0:-leftout]
                estY[k] = np.dot(lamb, np.transpose(realisation))
        else:
            raise TypeError("Die Variable 'order' muss eine Natuerliche Zahl oder -1 sein")

        return estY

    else:
        raise TypeError("Entweder die Kovarianzfunktion oder die generalisierten Kovarianzfunktion soll angegeben werden")

xachs = np.arange(0, 2, 0.01)
xLen = len(xachs)

# fluctation X erzeugen, X_t = W_t+h - W_t
h = 15
stDev = np.sqrt(0.01)
z = np.random.normal(0,stDev,size=xLen+h-1)
#X = np.append([0], np.cumsum(z))[0:xLen]
bm = np.append([0], np.cumsum(z))
X = bm[h:] - bm[0:xLen]

# drift m erzeugen
m = np.zeros(xLen)
for i in range(xLen):
    m[i] = f(xachs[i])

# Prozess Y = drift + fluctation
Y = m+X

# Punkte mit bekannten Werte
datapoints = np.arange(0, 2, 0.25)
ratio = len(xachs)/len(datapoints)
realisation = Y[::ratio]

# Kriging anwenden
estY3 = kriging(-1, datapoints, realisation, drift=[drift_1, drift_2], genCov=genCov )
estY1 = kriging(0, datapoints, realisation, covariancefunction, [drift_1, drift_2])
estY2 = kriging(-1, datapoints, realisation, covariancefunction, [drift_1, drift_2] )

# Ergebnis plotten 
plt.plot(datapoints, realisation, 'bo')
plt.plot(xachs,Y)
plt.plot(xachs, estY1, color = 'red')
plt.plot(xachs, estY2, color = 'blue')
plt.plot(xachs, estY3, color = 'green')

plt.show()

# Quadratische Abweichung berechnen
error1 = sum((estY1 - Y) * (estY1 - Y))
print(error1)
error2 = sum((estY2 - Y) * (estY2 - Y))
print(error2)
error3 = sum((estY3 - Y) * (estY3 - Y))
print(error3)


a = covariancefunction(0.02, 0.08, 0.15)
b = covariancefunction(0.10, 0.04, 0.15)
c = genCov(-0.06, 0.15)

print(a)
print(b)
print(c)
