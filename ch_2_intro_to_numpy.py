# %%
import numpy as np
import array

# %%
result = 0
for i in range(100):
    result += i

print(result)
# %%
L = list(range(10))
L

# %%
type(L[0])

# %%
L2 = [str(c) for c in L]
L2
# %%
type(L2[0])

# %%
L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]

# %%
L = list(range(10))
A = array.array("i", L)
A

# %%
np.array([1, 4, 2, 5, 3])

# %%
np.array([3.14, 4, 2, 3])

# %%
np.array([1, 2, 3, 4], dtype="float32")

# %%
np.array([range(i, i + 3) for i in [2, 4, 6]])

# %%
np.zeros(10, dtype=int)

# %%
np.ones((3, 5), dtype=float)

# %%
np.full((3, 5), 3.14)

# %%
np.arange(0, 20, 2)

# %%
np.linspace(0, 1, 5)

# %%
np.random.normal(0, 1, (3, 3))

# %%
np.random.randint(0, 10, (3, 3))

# %%
np.eye(3)

# %%
np.empty(3)

# %%
np.zeros(10, dtype="int16")

# %%
np.zeros(10, dtype=np.int16)

# %%
np.random.seed(0)

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))


# %%
print("x3 ndim: ", x3.ndim)
print("x3 shape: ", x3.shape)
print("x3 size: ", x3.size)
print(x3)
# %%
print("dtype:", x3.dtype)

# %%
print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")


# %%
x1

# %%
x1[0]

# %%
x1[4]

# %%
x1[-1]

# %%
x1[-2]

# %%
x2

# %%
x2[0, 0]

# %%
x2[1, 0]

# %%
x2[2, 1]

# %%
x2[0, 0] = 12
x2

# %%
x1[0] = 3.94159
x1

# %%
x = np.arange(10)
x

# %%
x[:5]

# %%
x[5:]

# %%
x[4:7]

# %%
x[::2]

# %%
x[1::2]

# %%
x[::-1]

# %%
x[5::-2]

# %%
x2

# %%
x2[:2, :3]

# %%
x2[:3, ::2]

# %%
x2[::-1, ::-1]

# %%
print(x2[:, 0])

# %%
print(x2[0, :])

# %%
print(x2[0])

# %%
print(x2)

# %%
x2_sub = x2[:2, :2]
print(x2_sub)

# %%
x2_sub[0, 0] = 99
print(x2_sub)

# %%
print(x2)

# %%
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)

# %%
x2_sub_copy[0, 0] = 42
print(x2_sub_copy)

# %%
print(x2)

# %%
grid = np.arange(1, 10).reshape((3, 3))
print(grid)

# %%
x = np.array([1, 2, 3])
x.reshape((1, 3))

# %%
x[np.newaxis, :]

# %%
x.reshape((3, 1))

# %%
x[:, np.newaxis]

# %%
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

# %%
z = [99, 99, 99]
print(np.concatenate([x, y, z]))

# %%
grid = np.array([[1, 2, 3], [4, 5, 6]])

# %%
np.concatenate([grid, grid])

# %%
np.concatenate([grid, grid], axis=1)

# %%
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7], [6, 5, 4]])

np.vstack([x, grid])
# %%
y = np.array([[99], [99]])
np.hstack([grid, y])

# %%
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

# %%
grid = np.arange(16).reshape((4, 4))
grid

# %%
upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

# %%
left, right = np.hsplit(grid, [2])
print(left)
print(right)

# %%
np.random.seed(0)


def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output


values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

# %%
print(compute_reciprocals(values))
print(1.0 / values)

# %%
np.arange(5) / np.arange(1, 6)


# %%
x = np.arange(9).reshape((3, 3,))
2 ** x

# %%
x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)


# %%
print("-x =", -x)
print("x ** 2 =", x ** 2)
print("x % 2 =", x % 2)

# %%
-((0.5 * x + 1) ** 2)

# %%
np.add(x, 2)

# %%
x = np.array([-2, -1, 0, 1, 2])
abs(x)

# %%
np.absolute(x)

# %%
np.abs(x)

# %%
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)

# %%
theta = np.linspace(0, np.pi, 3)

# %%
print("theta =", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

# %%
x = [-1, 0, 1]
print("x =", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

# %%
x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))

# %%
x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))

# %%
x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))

# %%
