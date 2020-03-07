# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil import parser

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
pd.Series([1, np.nan, 2, None])

# %%
x = pd.Series(range(2), dtype=int)
x

# %%
x[0] = None
x
# %%
data = pd.Series([1, np.nan, "hello", None])

# %%
data.isnull()

# %%
data[data.notnull()]

# %%
data.dropna()

# %%
df = pd.DataFrame([[1, np.nan, 2], [2, 3, 5], [np.nan, 4, 6]])
df

# %%
df.dropna()

# %%
df.dropna(axis="columns")

# %%
df[3] = np.nan
df

# %%
df.dropna(axis="columns", how="all")

# %%
df.dropna(axis="rows", thresh=3)

# %%
data = pd.Series([1, np.nan, 2, None, 3], index=list("abcde"))

# %%
data.fillna(0)

# %%
data.fillna(methods="ffill")

# %%
data.fillna(method="bfill")

# %%
df

# %%
df.fillna(method="ffill", axis=1)

# %%
index = [
    ("California", 2000),
    ("California", 2010),
    ("New York", 2000),
    ("New York", 2010),
    ("Texas", 2000),
    ("Texas", 2010),
]
populations = [33871648, 37253956, 18976457, 19378102, 20851820, 25145561]
pop = pd.Series(populations, index=index)
pop

# %%
pop[("California", 2010):("Texas", 2000)]

# %%
pop[[i for i in pop.index if i[1] == 2010]]

# %%
index = pd.MultiIndex.from_tuples(index)
index

# %%
pop = pop.reindex(index)
pop

# %%
pop[:, 2010]

# %%
pop_df = pop.unstack()
pop_df
# %%
pop_df.stack()

# %%
pop_df = pd.DataFrame(
    {"total": pop, "under18": [9267089, 92840494, 4687374, 4318033, 5906301, 6879014]}
)
pop_df
# %%
f_u18 = pop_df["under18"] / pop_df["total"]
f_u18.unstack()

# %%
df = pd.DataFrame(
    np.random.rand(4, 2),
    index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
    columns=["data1", "data2"],
)
df

# %%
data = {
    ("California", 2000): 33871648,
    ("California", 2010): 37253956,
    ("Texas", 2000): 20851820,
    ("Texas", 2010): 25145561,
    ("New York", 2000): 18976457,
    ("New York", 2010): 19378102,
}
pd.Series(data)

# %%
pd.MultiIndex.from_arrays([["a", "a", "b", "b"], [1, 2, 1, 2]])

# %%
pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])

# %%
pd.MultiIndex.from_product([["a", "b"], [1, 2]])

# %%
# pd.MultiIndex(levels=[['a','b'], [1,2]], labels=[[0,0,1,1], [0,1,0,1]])

# %%
pop.index.names = ["state", "year"]
pop

# %%
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=["year", "visit"])
columns = pd.MultiIndex.from_product(
    [["Bob", "Guido", "Sue"], ["HR", "Temp"]], names=["subject", "type"]
)

data = np.round(np.random.rand(4, 6), 1)
data[:, ::2] *= 10
data += 37

health_data = pd.DataFrame(data, index=index, columns=columns)
health_data

# %%
health_data["Guido"]

# %%
pop

# %%
pop["California", 2000]

# %%
pop["California", 2000]

# %%
pop["California"]

# %%
pop.loc["California":"New York"]

# %%
pop[:, 2000]

# %%
pop[pop > 2200000]

# %%
pop[["California", "Texas"]]

# %%
health_data

# %%
health_data["Guido", "HR"]

# %%
health_data.iloc[:2, :2]

# %%
health_data.loc[:, ("Bob", "HR")]

# %%

# health_data.loc[(:, 1), (:, 'HR')]
# %%
idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, "HR"]]

# %%
index = pd.MultiIndex.from_product([["a", "c", "b"], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ["char", "int"]
data

# %%
try:
    data["a":"b"]
except KeyError as e:
    print(type(e))
    print(e)

# %%
data = data.sort_index()
data

# %%
data["a":"b"]

# %%
pop.unstack(level=0)


# %%
pop.unstack(level=1)

# %%
pop.unstack(level=1)

# %%
pop.unstack().stack()

# %%
pop_flat = pop.reset_index(name="population")
pop_flat

# %%
pop_flat.set_index(["state", "year"])

# %%
health_data

# %%
data_mean = health_data.mean(level="year")
data_mean

# %%
data_mean.mean(axis=1, level="type")

# %%


def make_df(cols, ind):
    """Quickly make a DF"""
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)


make_df("ABC", range(3))

# %%
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])

# %%
x = [[1, 2], [3, 4]]
np.concatenate([x, x], axis=1)

# %%
ser1 = pd.Series(["A", "B", "C"], index=[1, 2, 3])
ser2 = pd.Series(["D", "E", "F"], index=[4, 5, 6])
pd.concat([ser1, ser2])

# %%
df1 = make_df("AB", [1, 2])
df2 = make_df("AB", [3, 4])
print(df1)
print(df2)
print(pd.concat([df1, df2]))

# %%
df3 = make_df("AB", [0, 1])
df4 = make_df("CD", [0, 1])
print(df3)
print(df4)
print(pd.concat([df3, df4], axis=1))

# %%
x = make_df("AB", [0, 1])
y = make_df("AB", [2, 3])
y.index = x.index
print(x)
print(y)
print(pd.concat([x, y]))

# %%
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError", e)


# %%
print(x)
print(y)
print(pd.concat([x, y], ignore_index=True))

# %%
print(pd.concat([x, y], keys=["x", "y"]))

# %%
df5 = make_df("ABC", [1, 2])
df6 = make_df("BCD", [3, 4])
print(df5)
print(df6)
print(pd.concat([df5, df6]))

# %%
print(pd.concat([df5, df6], join="inner"))

# %%
# print(pd.concat([df5, df6], join_axes=[df5.columns]))

# %%
print(df1.append(df2))


# %%
df1 = pd.DataFrame(
    {
        "employee": ["Bob", "Jake", "Lisa", "Sue"],
        "group": ["Accounting", "Engineering", "Engineering", "HR"],
    }
)
df2 = pd.DataFrame(
    {"employee": ["Lisa", "Bob", "Jake", "Sue"], "hire_date": [2004, 2008, 2012, 2014]}
)
print(df1)
print(df2)

# %%
df3 = pd.merge(df1, df2)
df3

# %%
df4 = pd.DataFrame(
    {
        "group": ["Accounting", "Engineering", "HR"],
        "supervisor": ["Carly", "Guido", "Steve"],
    }
)
print(pd.merge(df3, df4))

# %%
df5 = pd.DataFrame(
    {
        "group": ["Accounting", "Accounting", "Engineering", "Engineering", "HR", "HR"],
        "skills": [
            "math",
            "spreadsheets",
            "coding",
            "linux",
            "spreadsheets",
            "organization",
        ],
    }
)
print(df1)
print(df5)
print(pd.merge(df1, df5))

# %%
print(pd.merge(df1, df2, on="employee"))

# %%
df3 = pd.DataFrame(
    {"name": ["Bob", "Jake", "Lisa", "Sue"], "salary": [70000, 80000, 120000, 90000]}
)
print(df1)
print(df2)
print(pd.merge(df1, df3, left_on="employee", right_on="name"))

# %%
pd.merge(df1, df3, left_on="employee", right_on="name").drop("name", axis=1)

# %%
df1a = df1.set_index("employee")
df2a = df2.set_index("employee")
print(df1a)
print(df2a)

# %%
print(pd.merge(df1a, df2a, left_index=True, right_index=True))

# %%
print(df1a.join(df2a))

# %%
print(pd.merge(df1a, df3, left_index=True, right_on="name"))

# %%
df6 = pd.DataFrame(
    {"name": ["Peter", "Paul", "Mary"], "food": ["fish", "beans", "bread"]},
    columns=["name", "food"],
)
df7 = pd.DataFrame(
    {"name": ["Mary", "Jospeh"], "drink": ["wine", "beer"]}, columns=["name", "drink"]
)
print(pd.merge(df6, df7))

# %%
pd.merge(df6, df7, how="inner")

# %%
print(pd.merge(df6, df7, how="outer"))

# %%
print(pd.merge(df6, df7, how="left"))

# %%
df8 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"], "rank": [1, 2, 3, 4]})
df9 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"], "rank": [3, 1, 4, 2]})
print(pd.merge(df8, df9, on="name"))

# %%
print(pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]))

# %%
pop = pd.read_csv("data/state-population.csv")
areas = pd.read_csv("data/state-areas.csv")
abbrevs = pd.read_csv("data/state-abbrevs.csv")

# %%
merged = pd.merge(
    pop, abbrevs, how="outer", left_on="state/region", right_on="abbreviation"
)
merged = merged.drop("abbreviation", 1)
merged.head()

# %%
merged.isnull().any()

# %%
merged[merged["population"].isnull()].head()

# %%
merged.loc[merged["state"].isnull(), "state/region"].unique()

# %%
merged.loc[merged["state/region"] == "PR", "state"] = "Puerto Rico"
merged.loc[merged["state/region"] == "USA", "state"] = "United States"
merged.isnull().any()

# %%
final = pd.merge(merged, areas, on="state", how="left")
final.head()

# %%
final.isnull().any()

# %%
final["state"][final["area (sq. mi)"].isnull()].unique()


# %%
final.dropna(inplace=True)
final.head()

# %%
data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()
# %%
data2010.set_index("state", inplace=True)
density = data2010["population"] / data2010["area (sq. mi)"]


# %%
density.sort_values(ascending=False, inplace=True)
density.head()

# %%
density.tail()

# %%
planets = sns.load_dataset("planets")
planets.shape

# %%
planets.head()

# %%
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser

# %%
ser.sum()

# %%
ser.mean()

# %%
df = pd.DataFrame({"A": rng.rand(5), "B": rng.rand(5)})
df

# %%
df.mean()

# %%
df.mean(axis="columns")

# %%
planets.dropna().describe()

# %%
df = pd.DataFrame(
    {"key": ["A", "B", "C", "A", "B", "C"], "data": range(6)}, columns=["key", "data"]
)
df

# %%
df.groupby("key")

# %%
df.groupby("key").sum()

# %%
planets.groupby("method")

# %%
planets.groupby("method")["orbital_period"]

# %%
planets.groupby("method")["orbital_period"].median()

# %%
for (method, group) in planets.groupby("method"):
    print("{0:30s} shape={1}".format(method, group.shape))

# %%
planets.groupby("method")["year"].describe().unstack()

# %%
rng = np.random.RandomState(0)
df = pd.DataFrame(
    {
        "key": ["A", "B", "C", "A", "B", "C"],
        "data1": range(6),
        "data2": rng.randint(0, 10, 6),
    },
    columns=["key", "data1", "data2"],
)

df

# %%
df.groupby("key").aggregate(["min", np.median, max])

# %%
df.groupby("key").aggregate({"data1": "min", "data2": "max"})

# %%


def filter_func(x):
    return x["data2"].std() > 4


print(df.groupby("key").filter(filter_func))

# %%
df.groupby("key").transform(lambda x: x - x.mean())

# %%


def norm_by_data2(x):
    x["data1"] /= x["data2"].sum()
    return x


print(df.groupby("key").apply(norm_by_data2))

# %%
L = [0, 1, 0, 1, 2, 0]
print(df.groupby(L).sum())

# %%
print(df.groupby(df["key"]).sum())

# %%
df2 = df.set_index("key")
mapping = {"A": "vowel", "B": "consonant", "C": "consonant"}
print(df2.groupby(mapping).sum())

# %%
print(df2.groupby(str.lower).mean())

# %%
df2.groupby([str.lower, mapping]).mean()

# %%
decade = 10 * (planets["year"] // 10)
decade = decade.astype(str) + "s"
decade.name = "decade"
planets.groupby(["method", decade])["number"].sum().unstack().fillna(0)

# %%
titanic = sns.load_dataset("titanic")

# %%
titanic.head()

# %%
titanic.groupby("sex")[["survived"]].mean()

# %%
titanic.groupby(["sex", "class"])["survived"].aggregate("mean").unstack()

# %%
titanic.pivot_table("survived", index="sex", columns="class")

# %%
age = pd.cut(titanic["age"], [0, 18, 80])
titanic.pivot_table("survived", ["sex", age], "class")

# %%
fare = pd.qcut(titanic["fare"], 2)
titanic.pivot_table("survived", ["sex", age], [fare, "class"])

# %%
titanic.pivot_table(
    index="sex", columns="class", aggfunc={"survived": sum, "fare": "mean"}
)

# %%
titanic.pivot_table("survived", index="sex", columns="class", margins=True)

# %%
births = pd.read_csv("data/births.csv")
births.head()

# %%
births["decade"] = 10 * (births["year"] // 10)
births.pivot_table("births", index="decade", columns="gender", aggfunc="sum")

# %%
sns.set()
births.pivot_table("births", index="year", columns="gender", aggfunc="sum").plot()
plt.ylabel("total births per year")

# %%
quartiles = np.percentile(births["births"], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])

# %%
births = births.query("(births > @mu -5 * @sig) & (births < @mu + 5 * @sig)")

# %%
births["day"] = births["day"].astype(int)

# %%
births.index = pd.to_datetime(
    10000 * births.year + 100 * births.month + births.day, format="%Y%m%d"
)
births["dayofweek"] = births.index.dayofweek

# %%
births.pivot_table("births", index="dayofweek", columns="decade", aggfunc="mean").plot()
plt.gca().set_xticklabels(["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"])
plt.ylabel("mean births by day")

# %%
births_by_date = births.pivot_table("births", [births.index.month, births.index.day])
births_by_date.head()

# %%
births_by_date.index = [
    pd.datetime(2012, month, day) for (month, day) in births_by_date.index
]
births_by_date.head()

# %%
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# %%
x = np.array([2, 3, 5, 7, 11, 13])
x * 2

# %%
data = ["peter", "Paul", "MARY", "gUIDO"]
[s.capitalize() for s in data]

# %%
data = ["peter", "Paul", None, "MARY", "gUIDO"]
# %%
names = pd.Series(data)

# %%
names.str.capitalize()

# %%
monte = pd.Series(
    [
        "Graham Chapman",
        "John Cleese",
        "Terry Gilliam",
        "Eric Idle",
        "Terry Jones",
        "Michael Palin",
    ]
)


# %%
monte.str.lower()

# %%
monte.str.len()

# %%
monte.str.startswith("T")

# %%
monte.str.split()

# %%
monte.str.extract("([A-Za-z]+)")

# %%
monte.str.findall(r"^[^AEIOU].*[^aeiou]$")

# %%
monte.str[0:3]

# %%
monte.str.split().str.get(-1)

# %%
full_monte = pd.DataFrame(
    {"name": monte, "info": ["B|C|D", "B|D", "A|C", "B|D", "B|C", "B|C|D"]}
)
full_monte

# %%
full_monte["info"].str.get_dummies("|")

# %%
# Skipped recipie section- data file does not exist

# %%
datetime(year=2015, month=7, day=4)

# %%
date = parser.parse("4th of July, 2015")
date

# %%
date.strftime("%A")

# %%
date = np.array("2015-07-04", dtype=np.datetime64)
date

# %%
date + np.arange(12)

# %%
np.datetime64("2015-07-04")

# %%
np.datetime("2015-07-04 12:00")
