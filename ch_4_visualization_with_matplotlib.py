# %%
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

# %%
plt.style.use("classic")

# %%
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), "-")
plt.plot(x, np.cos(x), "--")
plt.show()

# %%
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))

# %%
fig, ax = plt.subplots(2)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

# %%
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))

# %%
plt.plot(x, np.sin(x))

# %%
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

# %%
plt.plot(x, np.sin(x - 0), color="blue")
plt.plot(x, np.sin(x - 1), color="g")
plt.plot(x, np.sin(x - 2), color=".75")
plt.plot(x, np.sin(x - 3), color="#FFDD44")
plt.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3))
plt.plot(x, np.sin(x - 5), color="chartreuse")

# %%
plt.plot(x, x + 0, linestyle="solid")
plt.plot(x, x + 1, linestyle="dashed")
plt.plot(x, x + 2, linestyle="dashdot")
plt.plot(x, x + 3, linestyle="dotted")

plt.plot(x, x + 4, linestyle="-")
plt.plot(x, x + 5, linestyle="--")
plt.plot(x, x + 6, linestyle="-.")
plt.plot(x, x + 7, linestyle=":")

# %%
plt.plot(x, x + 0, "-g")
plt.plot(x, x + 1, "--c")
plt.plot(x, x + 2, "-.k")
plt.plot(x, x + 3, ":r")

# %%
