from matplotlib import pyplot as plt
import numpy as np

# Methode um dem Kriging Schaetzer zu bestimmen
def kriging(order, datapoints, measurement, 
    covariancefunction=None, drift=None, genCov=None):
    '''Kriging Schaetzer von Y wird bestimmet

    gewoehnliche Kriging: kriging(order=-1, 
        datapoints, measurement, 
        covariancefunction)
    universale Kriging: kriging(order=-1, 
        datapoints, measurement, 
        covariancefunction, drift)
    intrinsische Kriging der ordnung k: 
        kriging(order=k, datapoints, 
        measurement, drift, genCov)

    arguments:
    order (int>=1) -- Y ist eine IRF von der Ordnung "order"
    datapoints (array) -- Messpunkten
    measurement (array) -- Messwerte von Y an den Messpunkten
    covariancefunction (function) -- Kovarianzfunktion 
        des deterministischen Teil von Y
    drift -- (array) Basen des Tendenzraumes
    genCov (function) -- generalisierte Kovarianzfunktion 
        des deterministischen Teil von Y

    return:
    estY (array) -- Kriging Schaetzer von Y
    '''
    # gewoehnliche Kriging
    if drift == None:
        left = np.zeros([len(datapoints)+1, 
                len(datapoints)+1])
    # universale Kriging
    if drift != None and order == -1:
        left = np.zeros([len(datapoints)+len(drift), 
                len(datapoints)+len(drift)])
    # intrinsische Kriging
    if isinstance(order, int) and order >= 0:
        left = np.zeros([len(datapoints)+len(drift) 
        +order+1, len(datapoints)+len(drift)+order+1])
    
    for i, data1 in enumerate(datapoints):
        for j, data2 in enumerate(datapoints):
            if covariancefunction != None:    
                left[i,j] = covariancefunction(data1, 
                        data2, 0.15)
            elif genCov != None:
                left[i,j] = genCov(data2-data1)
            else:
                raise TypeError("Entweder die Kovarianzfunktion\
                    oder die generalisierten Kovarianzfunktion\
                    soll angegeben werden")
    # gewoehnliche Kriging
    if drift == None:
        for m in range(len(datapoints)):
            left[len(datapoints) , m] = 1
            left[m, len(datapoints) ] = 1
    # universale Kriging
    if drift != None:
        for m, data in enumerate(datapoints):
            for n, d in enumerate(drift):
                left[m, len(datapoints)+n] = d(data)
                left[len(datapoints)+n, m] = d(data)
    if isinstance(order, int) and order >= 0:
        for m, data in enumerate(datapoints):
            for n in range(order+1):
                left[m, len(datapoints)+len(drift)+n] = data**n
                left[len(datapoints)+len(drift)+n, m] = data**n 

    estY = np.zeros(len(np.arange(0, 2, 0.01)))
    for k, t in enumerate(np.arange(0, 2, 0.01)):
        if drift == None:
            right = np.zeros(len(datapoints)+1)
        if drift != None and order == -1:
            right = np.zeros(len(datapoints)+len(drift))
        if isinstance(order, int) and order >= 0:
            right = np.zeros(len(datapoints)+len(drift)+order+1)
        
        for o, data in enumerate(datapoints):
            if covariancefunction != None:    
                right[o] = covariancefunction(data, t)
            elif genCov != None:    
                right[o] = genCov(t-data)
            else:
                raise TypeError("Entweder die Kovarianzfunktion\
                    oder die generalisierten Kovarianzfunktion\
                    soll angegeben werden")
        # gewoehnliche Kriging
        if drift == None:
            right[len(datapoints)] = 1
            leftout =  1
        # universale Kriging
        if drift != None:
            for p, d in enumerate(drift):
                right[len(datapoints)+p] = d(t)
            leftout = len(drift) 
        # intrinsische Kriging
        if isinstance(order, int) and order >= 0:
            for q in range(order+1):
                right[len(datapoints)+len(drift)+q] = t**q
            leftout = len(drift)+order+1
        
        # lineares Gleichungssystem loesen
        lsg = np.linalg.solve(left, right)
        lamb = lsg[0:-leftout]
        estY[k] = np.dot(lamb, np.transpose(measurement))

    return estY
    
###########
#Beispiele#
###########

# Basen des Tendenzraumes
def drift_1(x):
    return np.sin(2*x)
def drift_2(x):
    return np.cos(x)

# fast konstante Tendenz 
def d_1(x):
    return 0.1*drift_2(x)+0.2*drift_1(x)
# nicht konstante Tendenz
def d_2(x):
    return 1*drift_2(x)+2*drift_1(x)

# Kovarianzfunktion von X_1
def covariancefunction_SRF(t,s,h=0.15):
    return 2*(t<=s)*t+2*(t>s)*s-(t<=s+h)*t-\
        (s+h<t)*(s+h)-(t+h<=s)*(t+h)-(s<t+h)*s+h

# Generalisierte Kovarianzfunktion von X_2
def genCov_IRF_0(d): 
    return 0.5*(d>0)*(-d)+0.5*(d<=0)*d


# X-Achse definieren
xachs = np.arange(0, 2, 0.01)
xLen = len(xachs)

# Fluktation X_1 erzeugen mit X_1_t = W_{t+0.15}-W_t
h = 15
stDev = np.sqrt(0.01)
z = np.random.normal(0,stDev,size=xLen+h-1)
# Brownianmotion von W_0 bis W_2.15 werden erzeugt
bm = np.append([0], np.cumsum(z)) 
X_1 = bm[h:]-bm[0:xLen]

# Fluktation X_2 erzeugen mit X_2_t = W_t
X_2 = np.append([0], np.cumsum(z))[0:xLen]

# Tendenz D_1 erzeugen
D_1 = np.zeros(xLen)
for i in range(xLen):
    D_1[i] = d_1(xachs[i])

# Tendenz D_2 erzeugen
D_2 = np.zeros(xLen)
for i in range(xLen):
    D_2[i] = d_2(xachs[i])

# Punkte mit bekannten Werte
datapoints = np.arange(0, 2, 0.25)
ratio = len(xachs)/len(datapoints)


## Simulation Y_1
print("\nSimulation Y_1")
Y_1 = D_1+X_1
measurement_1 = Y_1[::ratio]
estY1_1 = kriging(-1, datapoints, measurement_1, 
    covariancefunction=covariancefunction_SRF)
estY1_2 = kriging(-1, datapoints, measurement_1, 
    covariancefunction=covariancefunction_SRF, 
    drift=[drift_1, drift_2])
# Quadratische Abweichung berechnen
print("Quadratische Abweichungen:")
error1_1 = sum((estY1_1-Y_1)*(estY1_1-Y_1))
print("Gewoehnliche Kriging: "+str(error1_1))
error1_2 = sum((estY1_2-Y_1)*(estY1_2-Y_1))
print("Universale Kriging: "+str(error1_2))
# Ergebnis plotten 
plt.plot(datapoints, measurement_1, 'bo')
plt.xlim(0, 2)
plt.ylim(-2, 2)
plt.plot(xachs,D_1, color='orange', label='Tendenz')
plt.plot(xachs,Y_1, color='black')
plt.plot(xachs, estY1_1, color='red', 
    label='gewoehnliche Kriging')
plt.plot(xachs, estY1_2, color='green', 
    label='universale Kriging')
plt.legend(loc='upper right')
plt.show()


## Simulation Y_2
print("\nSimulation Y_2")
Y_2 = D_2+X_1
measurement_2 = Y_2[::ratio]
estY2_1 = kriging(-1, datapoints, measurement_2, 
    covariancefunction=covariancefunction_SRF)
estY2_2 = kriging(-1, datapoints, measurement_2, 
    covariancefunction=covariancefunction_SRF, 
    drift=[drift_1, drift_2])
# Quadratische Abweichung berechnen
print("Quadratische Abweichungen:")
error2_1 = sum((estY2_1-Y_2)*(estY2_1-Y_2))
print("Gewoehnliche Kriging: "+str(error2_1))
error2_2 = sum((estY2_2-Y_2)*(estY2_2-Y_2))
print("Universale Kriging: "+str(error2_2))
# Ergebnis plotten 
plt.plot(datapoints, measurement_2, 'bo')
plt.xlim(0, 2)
plt.ylim(-4, 4)
plt.plot(xachs,D_2, color='orange', label='Tendenz')
plt.plot(xachs,Y_2, color='black')
plt.plot(xachs, estY2_1, color='red', 
    label='gewoehnliche Kriging')
plt.plot(xachs, estY2_2, color='green', 
    label='universale Kriging')
plt.legend(loc='upper right')
plt.show()


## Simulation Y_3
print("\nSimulation Y_3")
Y_3 = D_2+X_2
measurement_3 = Y_3[::ratio]
estY3_1 = kriging(-1, datapoints, measurement_3, 
    covariancefunction=covariancefunction_SRF)
estY3_2 = kriging(-1, datapoints, measurement_3, 
    covariancefunction=covariancefunction_SRF, 
    drift=[drift_1, drift_2])
estY3_3 = kriging(0, datapoints, measurement_3, 
    drift=[drift_1, drift_2], genCov=genCov_IRF_0)
# Quadratische Abweichung berechnen
print("Quadratische Abweichungen:")
error3_1 = sum((estY3_1-Y_3)*(estY3_1-Y_3))
print("Gewoehnliche Kriging: "+str(error3_1))
error3_2 = sum((estY3_2-Y_3)*(estY3_2-Y_3))
print("Universale Kriging: "+str(error3_2))
error3_3 = sum((estY3_3-Y_3)*(estY3_3-Y_3))
print("Intrinsische Kriging: "+str(error3_3))
# Ergebnis plotten 
plt.plot(datapoints, measurement_3, 'bo')
plt.xlim(0, 2)
plt.ylim(-4, 4)
plt.plot(xachs,D_2, color='orange', label='Tendenz')
plt.plot(xachs,Y_3, color='black',)
plt.plot(xachs, estY3_1, color='red', 
    label='gewoehnliche Kriging')
plt.plot(xachs, estY3_2, color='green', 
    label='universale Kriging')
plt.plot(xachs, estY3_3, color='blue', 
    label='intrinsische Kriging')
plt.legend(loc='upper right')
plt.show()
