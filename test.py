from matplotlib import pyplot as plt
import numpy as np

def f(x):
    #return 3
    return 1* np.cos(x) + 2 * np.sin(2*x)
    #return 0.01* np.cos(x) + 0.02 * np.sin(x)

def covariancefunction(t,s,h):
    #return (t<=s) * t + (s<t) * s
    return 2 * (t<=s) * t + 2 * (t>s) * s  - (t<=s+h) * t - \
        (s+h<t) * (s+h) - (t+h<=s) * (t+h) - (s<t+h) * s + h

def genCov(d, h): 
    #return 0.5*(d>0)*(-d) + 0.5*(d<=0)*d
    if d>=0:
        return h - (h<=d) * h - (d<h) * d
    elif d<0:
        return h + d - (h+d) * (h+d<0)

def drift_1(x):
    return np.sin(2*x)

def drift_2(x):
    return np.cos(x)


def kriging(order, datapoints, realisation, 
    covariancefunction=None, drift=None, genCov=None):

    if drift == None:
        left = np.zeros([len(datapoints) + 1, 
                len(datapoints) + 1])
    if drift != None and order == -1:
        left = np.zeros([len(datapoints) + len(drift), 
                len(datapoints) + len(drift)])
    if isinstance(order, int) and order >= 0:
        left = np.zeros([len(datapoints) + len(drift) 
        + order + 1, len(datapoints) + len(drift) + order + 1])
    
    for i, data1 in enumerate(datapoints):
        for j, data2 in enumerate(datapoints):
            if covariancefunction != None:    
                left[i,j] = covariancefunction(data1, 
                        data2, 0.15)
            elif genCov != None:
                left[i,j] = genCov(data2-data1, 0.15)
            else:
                raise TypeError("Entweder die Kovarianzfunktion \
                    oder die generalisierten Kovarianzfunktion \
                    soll angegeben werden")

    if drift == None:
        for m in range(len(datapoints)):
            left[len(datapoints) , m] = 1
            left[m, len(datapoints) ] = 1
    if drift != None:
        for m, data in enumerate(datapoints):
            for n, d in enumerate(drift):
                left[m, len(datapoints) + n] = d(data)
                left[len(datapoints) + n, m] = d(data)
    if isinstance(order, int) and order >= 0:
        for m, data in enumerate(datapoints):
            for n in range(order+1):
                left[m, len(datapoints)+len(drift)+n] = data**n
                left[len(datapoints)+len(drift)+n, m] = data**n 

    estY = np.zeros(len(np.arange(0, 2, 0.01)))
    for k, t in enumerate(np.arange(0, 2, 0.01)):
        if drift == None:
            right = np.zeros(len(datapoints) + 1)
        if drift != None and order == -1:
            right = np.zeros(len(datapoints) + len(drift))
        if isinstance(order, int) and order >= 0:
            right = np.zeros(len(datapoints)+len(drift)+order+1)
        
        for o, data in enumerate(datapoints):
            if covariancefunction != None:    
                right[o] = covariancefunction(data, t, 0.15)
            elif genCov != None:    
                right[o] = genCov(t-data, 0.15)
            else:
                raise TypeError("Entweder die Kovarianzfunktion \
                    oder die generalisierten Kovarianzfunktion \
                    soll angegeben werden")

        if drift == None:
            right[len(datapoints)] = 1
            leftout =  1
        if drift != None:
            for p, d in enumerate(drift):
                right[len(datapoints) + p] = d(t)
            leftout = len(drift) 
        if isinstance(order, int) and order >= 0:
            for q in range(order+1):
                right[len(datapoints) + len(drift) + q] = t**q
            leftout = len(drift) + order + 1
        
        lsg = np.linalg.solve(left, right)
        lamb = lsg[0:-leftout]
        estY[k] = np.dot(lamb, np.transpose(realisation))

    return estY
    

 

xachs = np.arange(0, 2, 0.01)
xLen = len(xachs)

# Fluktation X erzeugen, X_t = W_t+h - W_t
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

# Prozess Y = drift + Fluktation
Y = m+X

# Punkte mit bekannten Werte
datapoints = np.arange(0, 2, 0.25)
ratio = len(xachs)/len(datapoints)
realisation = Y[::ratio]

# Kriging anwenden
estY1 = kriging(-1, datapoints, realisation, 
    covariancefunction=covariancefunction)
estY2 = kriging(-1, datapoints, realisation, 
    covariancefunction=covariancefunction, 
    drift=[drift_1, drift_2])
estY3 = kriging(-1, datapoints, realisation, 
    drift=[drift_1, drift_2], genCov=genCov )
estY4 = kriging(0, datapoints, realisation, 
    drift=[drift_1, drift_2], genCov=genCov )
estY5 = kriging(-1, datapoints, realisation, genCov=genCov )


# Quadratische Abweichung berechnen
error1 = sum((estY1 - Y) * (estY1 - Y))
print(error1)
error2 = sum((estY2 - Y) * (estY2 - Y))
print(error2)
error3 = sum((estY3 - Y) * (estY3 - Y))
print(error3)
error4 = sum((estY4 - Y) * (estY4 - Y))
print(error4)
error5 = sum((estY5 - Y) * (estY5 - Y))
print(error5)

# Ergebnis plotten 
plt.plot(datapoints, realisation, 'bo')
plt.plot(xachs,Y)
plt.plot(xachs, estY1, color = 'red')
plt.plot(xachs, estY2, color = 'blue')
plt.plot(xachs, estY3, color = 'green')
plt.plot(xachs, estY4, color = 'yellow')
plt.plot(xachs, estY5, color = 'black')
plt.show()

