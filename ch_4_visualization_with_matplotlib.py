# %% 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

# %% 
plt.style.use('classic')

# %%
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')
plt.show()