import numpy as np
import matplotlib.pyplot as plt

#def evolve(TotT,t, I):
#    TotT = TotT + t[I]
#    t = t - t[I]
#    t[I] = rd.expovariate(1)
#    return TotT, t

def get_data(results):
    Y = results
    N = len(Y)
    Mean = (1.0/N) * sum(Y)
    X = Y - Mean
    X2 = X ** 2
    Var = 1.0/(N-1) * sum(X2)
    stdev = np.sqrt(Var)
    sigma = np.sqrt(Var)
    X3 = (X/sigma) ** 3
    Skew = (1.0/N) * sum(X3)
    X4 = (X/sigma) ** 4
    Kurt = (1.0/N) * sum(X4) - 3

    return Mean, stdev, Skew, Kurt

def asep_sample():
    gamma = .20
    #gamma = 1.0-2.0*p                       # (q-p) in ASEP
    p = (0.5) * (1-gamma)
    N = 30                                  # Number of particles
    TotT = 0.0                               # Initialize master clock

    y = np.arange(0, N, step=1, dtype=int)   # Step Initial Condition
    t = np.random.exponential(1.0, N)        # Initialize particle clocks
    
    while( y[N-5] == N-5 ): # Keeps running until last five particles remain
    #EndT = 100.0                             # End time
    #while (TotT < EndT):
        Prob = np.random.uniform()
        I = t.argmin()
        # All possible ways for a selected particle to jump right w.p. p
        if (Prob < p) and (
            I > 0 and I+1 < N and y[I]+1 != y[I+1] # in between
            #or I+1 == N                             # lagg part.
            or I == 0  and y[I]+1 != y[I+1]         # lead part.
            ):
            y[I] = y[I]+1
        # All possible ways for a selected particle to jump left w.p. q
        elif (Prob > p) and (
            I > 0 and I+1 < N  and y[I]-1 != y[I-1] # in between jump
            or I == 0                                # lead part.
            #or I+1 == N and y[I]-1 != y[I-1]        # lagg part.
            ):
            y[I] = y[I]-1
        else: 
            pass

        TotT = TotT + t[I]                   # shift the master clock
        t = t - t[I]                         # let the other clocks pass
        t[I] = np.random.exponential(1) 
    # gamma > 0 for q > p
    scale = (gamma)* (TotT ** (1.0 / 2.0) )              # standard deviation 
    if gamma != 0.0:
        mean = gamma * TotT                      # mean for central limit theorem
        sample = (y[0]+mean)/scale
    else:
        mean =( np.log(TotT) ) ** (1.0/2.0) 
        sample = y[0]/scale + mean-1

    #return (sample, y[0], TotT)
    return (sample, y, TotT)

if __name__ == "__main__":
    CLT = [0]
    EndT = [0]
    r = 5
    runs = 1 * (10 ** r)                     # Number of asep samples
    sample = 0.0 
    x = [np.zeros(1, dtype = int)]
    #P = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40]

    for i in range(runs):
        (sample, y, TotT) = asep_sample()
        x.append(y)
        CLT.append(sample)
        EndT.append(TotT)
    
    del x[0]
    del CLT[0]
    del EndT[0]
    avgT = sum(EndT)/len(EndT)
    if r == 0:
        print(x)
        print(CLT)
    else:
        plt.figure()
        plt.hist(CLT,40) # runs = 50,000. N=20,EndT=200, use 57 bins.
        plt.show()
        VAL = get_data(CLT)
        print VAL 
        print avgT
        #np.savetxt('CLT_p=0.txt.', CLT, delimiter=',')
