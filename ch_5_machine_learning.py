# %%
import seaborn as sns

# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits

# %%
iris = sns.load_dataset("iris")
iris.head()

# %%
sns.pairplot(iris, hue="species", size=1.5)

# %%
X_iris = iris.drop("species", axis=1)
X_iris.shape

# %%
y_iris = iris["species"]
y_iris.shape

# %%
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)

# %%
model = LinearRegression(fit_intercept=True)
model

# %%
X = x[:, np.newaxis]
X.shape

# %%
model.fit(X, y)

# %%
model.coef_

# %%
model.intercept_

# %%
xfit = np.linspace(-1, 11)

# %%
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# %%
plt.scatter(x, y)
plt.plot(xfit, yfit)

# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

# %%
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

# %%
accuracy_score(ytest, y_model)

# %%
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)

# %%
iris["PCA1"] = X_2D[:, 0]
iris["PCA2"] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue="species", data=iris, fit_reg=False)

# %%
model = GaussianMixture(n_components=3, covariance_type="full")
model.fit(X_iris)
y_gmm = model.predict(X_iris)

# %%
iris["cluster"] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue="species", col="cluster", fit_reg=False)

# %%
digits = load_digits()
digits.images.shape

# %%
# fig, axes = plt.subplots(10,10, figsize=(8,8), subplot_kw={'xticks':[]})
