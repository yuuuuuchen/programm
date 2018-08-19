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
                        data2)
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
    

