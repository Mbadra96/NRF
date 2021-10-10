import numpy as np
from scipy import signal # type: ignore
from scipy.stats import norm # type: ignore
import matplotlib.pyplot as plt #type: ignore

hsa_window = [12, 15, 12]
hsa_fir = list()
hsa_fir.append(signal.triang(hsa_window[0]))
hsa_fir.append(norm.pdf(np.linspace(1, hsa_window[1], hsa_window[1]), 0, 5))
hsa_fir.append(signal.triang(hsa_window[2]))

hsa_m_thresholds = [0.85, 0.05, 0.5]

bsa_window = [9, 10, 8]
bsa_fir = list()
bsa_fir.append(signal.triang(bsa_window[0]))
bsa_fir.append(norm.pdf(np.linspace(1, bsa_window[1], bsa_window[1]), 1.5, 3.5))
bsa_fir.append(signal.triang(bsa_window[2]))


plt.plot(bsa_fir[0])
plt.plot(bsa_fir[1])
plt.plot(bsa_fir[2])
plt.show()