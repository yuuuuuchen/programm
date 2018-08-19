from matplotlib import pyplot as plt
import numpy as np
from Kriging import kriging

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

# Kovarianzfunktion von X1
def covariancefunction_X1(t,s,h=0.15):
    return 2*(t<=s)*t+2*(t>s)*s-(t<=s+h)*t-\
        (s+h<t)*(s+h)-(t+h<=s)*(t+h)-(s<t+h)*s+h
# Kovarianzfunktion von X2
def covariancefunction_X2(t,s):
    return (t<=s) * t + (s<t) * s
# verallgemeinerte Kovarianzfunktion von X2
def genCov_X2(d): 
    return 0.5*(d>0)*(-d)+0.5*(d<=0)*d


# X-Achse definieren
xachs = np.arange(0, 2, 0.01)
xLen = len(xachs)

# Fluktation X1 erzeugen mit X1_t = W_{t+0.15}-W_t
h = 15
stDev = np.sqrt(0.01)
z = np.random.normal(0,stDev,size=xLen+h-1)
# Brownianmotion von W_0 bis W_2.15 werden erzeugt
bm = np.append([0], np.cumsum(z)) 
X1 = bm[h:]-bm[0:xLen]

# Fluktation X2 erzeugen mit X2_t = W_t
X2 = np.append([0], np.cumsum(z))[0:xLen]

# Tendenz D1 erzeugen
D1 = np.zeros(xLen)
for i in range(xLen):
    D1[i] = d_1(xachs[i])

# Tendenz D2 erzeugen
D2 = np.zeros(xLen)
for i in range(xLen):
    D2[i] = d_2(xachs[i])

# Punkte mit bekannten Werte
datapoints = np.arange(0, 2, 0.25)
ratio = len(xachs)/len(datapoints)


## Simulation Y1
print("\nSimulation Y1")
Y1 = D1+X1
measurement_1 = Y1[::int(ratio)]
estY1_1 = kriging(-1, datapoints, measurement_1, 
    covariancefunction=covariancefunction_X1)
estY1_2 = kriging(-1, datapoints, measurement_1, 
    covariancefunction=covariancefunction_X1, 
    drift=[drift_1, drift_2])
# Quadratische Abweichung berechnen
print("Quadratische Abweichungen:")
error1_1 = sum((estY1_1-Y1)*(estY1_1-Y1))
print("Gewoehnliche Kriging: "+str(error1_1))
error1_2 = sum((estY1_2-Y1)*(estY1_2-Y1))
print("Universale Kriging: "+str(error1_2))
# Ergebnis plotten 
plt.plot(datapoints, measurement_1, 'bo', label='Messpunkten')
plt.xlim(0, 2)
plt.ylim(-4, 4)
plt.plot(xachs,D1, color='orange', label='Tendenz')
plt.plot(xachs,Y1, color='black', label='Y1')
plt.plot(xachs, estY1_1, color='red', 
    label='gewoehnliche Kriging')
plt.plot(xachs, estY1_2, color='green', 
    label='universale Kriging')
plt.legend(loc='lower left')
plt.show()


## Simulation Y2
print("\nSimulation Y2")
Y2 = D2+X1
measurement_2 = Y2[::int(ratio)]
estY2_1 = kriging(-1, datapoints, measurement_2, 
    covariancefunction=covariancefunction_X1)
estY2_2 = kriging(-1, datapoints, measurement_2, 
    covariancefunction=covariancefunction_X1, 
    drift=[drift_1, drift_2])
# Quadratische Abweichung berechnen
print("Quadratische Abweichungen:")
error2_1 = sum((estY2_1-Y2)*(estY2_1-Y2))
print("Gewoehnliche Kriging: "+str(error2_1))
error2_2 = sum((estY2_2-Y2)*(estY2_2-Y2))
print("Universale Kriging: "+str(error2_2))
# Ergebnis plotten 
plt.plot(datapoints, measurement_2, 'bo', label='Messpunkten')
plt.xlim(0, 2)
plt.ylim(-4, 4)
plt.plot(xachs,D2, color='orange', label='Tendenz')
plt.plot(xachs,Y2, color='black', label='Y2')
plt.plot(xachs, estY2_1, color='red', 
    label='gewoehnliche Kriging')
plt.plot(xachs, estY2_2, color='green', 
    label='universale Kriging')
plt.legend(loc='lower left')
plt.show()


## Simulation Y3
print("\nSimulation Y3")
Y3 = D2+X2
measurement_3 = Y3[::int(ratio)]
estY3_1 = kriging(0, datapoints, measurement_3, 
    drift=[drift_1, drift_2], genCov=genCov_X2)
estY3_2 = kriging(-1, datapoints, measurement_3, 
    covariancefunction=covariancefunction_X2,
    drift=[drift_1, drift_2])
estY3_3 = kriging(-1, datapoints, measurement_3, 
    drift=[drift_1, drift_2], genCov=genCov_X2)

# Quadratische Abweichung berechnen
print("Quadratische Abweichungen:")
error3_1 = sum((estY3_1-Y3)*(estY3_1-Y3))
print("Intrinsische Kriging: "+str(error3_1))
error3_2 = sum((estY3_2-Y3)*(estY3_2-Y3))
print("Universale Kriging mit Kovarianzfunktion: "
    +str(error3_2))
error3_3 = sum((estY3_3-Y3)*(estY3_3-Y3))
print("Universale Kriging mit verallgemeinerter \
    Kovarianzfunktion: "+str(error3_3))

# Ergebnis plotten 
plt.plot(datapoints, measurement_3, 'bo', label='Messpunkten')
plt.xlim(0, 2)
plt.ylim(-5, 5)
plt.plot(xachs,D2, color='orange', label='Tendenz')
plt.plot(xachs,Y3, color='black', label='Y3')
plt.plot(xachs, estY3_1, color='purple', 
    label='intrinsische Kriging')
plt.plot(xachs, estY3_2, color='green', 
    label='universale Kriging mit Kovarianzfunktion')
plt.plot(xachs, estY3_3, color='blue', 
    label='universale Kriging mit verallgemeinerter \
     Kovarianzfunktion')

plt.legend(loc='lower left')
plt.show()