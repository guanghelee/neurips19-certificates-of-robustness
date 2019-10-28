import numpy as np
import sys

p = float(sys.argv[1])
print('p =', p)
std = np.sqrt(p * (1-p))
print('std =', std)
