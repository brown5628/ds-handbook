# %%
import numpy as np
import pandas as pd

# %%
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

# %%
data.values

# %%
data.index

# %%
data[1]

# %%
data[1:3]

# %%
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d"])
data

# %%
data["b"]

# %%
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])
data

# %%
data[5]

# %%
population_dict = {
    "California": 38332521,
    "Texas": 26448193,
    "New York": 19651127,
    "Florida": 19552860,
    "Illinois": 12882135,
}
population = pd.Series(population_dict)
population

# %%
population["California"]

# %%
population["California":"Illinois"]

# %%
pd.Series([2, 4, 6])

# %%
pd.Series(5, index=[100, 200, 300])

# %%
pd.Series({2: "a", 1: "b", 3: "c"})

# %%
pd.Series({2: "a", 1: "b", 3: "c"}, index=[3, 2])

# %%
area_dict = {
    "California": 423967,
    "Texas": 695662,
    "New York": 141297,
    "Florida": 170312,
    "Illinois": 149995,
}
area = pd.Series(area_dict)
area

# %%
states = pd.DataFrame({"population": population, "area": area})
states

# %%
states.index

# %%
states.columns

# %%
states["area"]

# %%
pd.DataFrame(population, columns=["population"])

# %%
data = [{"a": i, "b": 2 * i} for i in range(3)]
pd.DataFrame(data)

# %%
pd.DataFrame([{"a": 1, "b": 2}, {"b": 3, "c": 4}])

# %%
pd.DataFrame({"population": population, "area": area})

# %%
pd.DataFrame(np.random.rand(3, 2), columns=["foo", "bar"], index=["a", "b", "c"])

# %%
A = np.zeros(3, dtype=[("A", "i8"), ("B", "f8")])
A

# %%
pd.DataFrame(A)

# %%
ind = pd.Index([2, 3, 5, 7, 11])
ind

# %%
ind[1]

# %%
ind[::2]

# %%
print(ind.size, ind.shape, ind.ndim, ind.dtype)

# %%
ind[1] = 0

# %%
indA = pd.Index([1, 3, 5, 6, 9])
indB = pd.Index([2, 3, 5, 7, 11])


# %%
indA & indB

# %%
indA | indB

# %%
indA ^ indB

# %%
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d"])
data

# %%
data["b"]

# %%
"a" in data

# %%
data.keys()

# %%
list(data.items())

# %%
data["e"] = 1.25

# %%
data["a":"c"]

# %%
data[0:2]

# %%
data[(data > 0.3) & (data < 0.8)]

# %%
data[["a", "e"]]

# %%
data = pd.Series(["a", "b", "c"], index=[1, 3, 5])
data

# %%
data[1]

# %%
data[1:3]

# %%
data.loc[1]

# %%
data.loc[1:3]

# %%
data.iloc[1]

# %%
data.iloc[1:3]

# %%
area

# %%
population

# %%
data = pd.DataFrame({"area": area, "pop": population})
data

# %%
data["area"]


# %%
data.area

# %%
data.area is data["area"]

# %%
data.pop is data["pop"]

# %%
data["density"] = data["pop"] / data["area"]
data

# %%
data.values

# %%
data.T

# %%
data.values[0]

# %%
data["area"]

# %%
data.iloc[:3, :2]

# %%
data.loc[:"Illinois", :"pop"]

# %%
data.ix[:3, :"pop"]

# %%
data.loc[data.density > 100, ["pop", "density"]]

# %%
data.iloc[0, 2] = 90

# %%
data["Florida":"Illinois"]

# %%
data[1:3]

# %%
data[data.density > 100]

# %%
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser

# %%
df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=["A", "B", "C", "D"])
df

# %%
np.exp(ser)

# %%
np.sin(df * np.pi / 4)

# %%
area = pd.Series({"Alaska": 172337, "Texas": 695662, "California": 423967}, name="area")
population = pd.Series(
    {"California": 38332521, "Texas": 26448193, "New York": 19651127}, name="population"
)

# %%
population / area

# %%
area.index | population.index

# %%
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B

# %%
A.add(B, fill_value=0)

# %%
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list("AB"))
A

# %%
B = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=list("BAC"))
B

# %%
A + B

# %%
fill = A.stack().mean()
A.add(B, fill_value=fill)

# %%
A = rng.randint(10, size=(3, 4))
A

# %%
A - A[0]

# %%
df = pd.DataFrame(A, columns=list("QRST"))
df - df.iloc[0]

# %%
df.subtract(df["R"], axis=0)

# %%
halfrow = df.iloc[0, ::2]
halfrow

# %%
df - halfrow

# %%
vals1 = np.array([1, None, 3, 4])
vals1

# %%
vals1.sum()

# %%
vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype

# %%
1 + np.nan

# %%
0 * np.nan

# %%
vals2.sum(), vals2.min(), vals2.max()

# %%
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

# %%
