from scipy.special import gamma
import numpy as np

def volumeball(d,R):
    return np.pi**(d/2)/gamma(d/2+1)*R**d
def areasphere(d,R):
    return d*np.pi**(d/2)/gamma(d/2+1)*R**(d-1)

def Nbd(N_int,d):
    return N_int ** (1-1/d) * areasphere(d,1)/volumeball(d-1,1)

for d in range(2,6):
    print(Nbd(2000,d))
