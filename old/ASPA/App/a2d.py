import numpy as np
import matplotlib.pyplot as plt

# Sampling
fs = 100 
T = 1/fs  
t = np.arange(0, 1, T) 

# Analog signal  
f = 5
x = np.sin(2*np.pi*f*t)

# Quantization
Q = 2**1
xq = np.round(x*Q)/Q

# Encoding
xb = np.zeros(len(xq),dtype=int)
for i in range(len(xq)):
  xb[i] = int(xq[i]*(Q-1)) 

# Plot    
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.ylabel('Analog signal')

plt.subplot(2, 1, 2)
plt.stem(t, xb)
plt.ylabel('Digital signal')
plt.xlabel('Time')

plt.tight_layout()
plt.show()