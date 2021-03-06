# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skimage.data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import nan
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_circles
from mpl_toolkits import mplot3d
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from matplotlib.image import imread
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from matplotlib import offsetbox
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from scipy.stats import mode
from sklearn.manifold import TSNE
from sklearn.datasets import load_sample_image
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
#from sklearn.datasets import fetch_species_distributions
# from mpl_toolkits import Basemap
# from sklearn.datasets.species_distributions import construct_grids
from sklearn.base import ClassifierMixin
from skimage import data, color, feature
from sklearn.feature_extraction.image import PatchExtractor
from itertools import chain

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
fig, axes = plt.subplots(
    10,
    10,
    figsize=(8, 8),
    subplot_kw={"xticks": [], "yticks": []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1),
)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
    ax.text(0.5, 0.5, str(digits.target[i]), transform=ax.transAxes, color="green")


# %%
X = digits.data
X.shape

# %%
y = digits.target
y.shape

# %%
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape

# %%
plt.scatter(
    data_projected[:, 0],
    data_projected[:, 1],
    c=digits.target,
    edgecolor="none",
    alpha=0.5,
    cmap=plt.cm.get_cmap("icefire", 10),
)
plt.colorbar(label="digit label", ticks=range(10))
plt.clim(-0.5, 9.5)

# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# %%
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

# %%
accuracy_score(ytest, y_model)

# %%
mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel("predicted value")
plt.ylabel("true value")

# %%
fig, axes = plt.subplots(
    10,
    10,
    figsize=(8, 8),
    subplot_kw={"xticks": [], "yticks": []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1),
)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
    ax.text(
        0.5,
        0.5,
        str(y_model[i]),
        transform=ax.transAxes,
        color="green" if (ytest[i] == y_model[i]) else "red",
    )

# %%
iris = load_iris()
X = iris.data
y = iris.target

# %%
model = KNeighborsClassifier(n_neighbors=1)


# %%
model.fit(X, y)
y_model = model.predict(X)

# %%
accuracy_score(y, y_model)

# %%
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

model.fit(X1, y1)

y2_model = model.predict(X2)
accuracy_score(y2, y2_model)

# %%
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)

# %%
cross_val_score(model, X, y, cv=5)

# %%
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
scores

# %%
scores.mean()

# %%


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# %%
def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1.0 / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)

# %%
X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color="black")
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label="degree={0}".format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc="best")

# %%
degree = np.arange(0, 21)
train_score, val_score = validation_curve(
    PolynomialRegression(), X, y, "polynomialfeatures__degree", degree, cv=7
)

plt.plot(degree, np.median(train_score, 1), color="blue", label="training score")
plt.plot(degree, np.median(val_score, 1), color="red", label="validation score")
plt.legend(loc="best")
plt.ylim(0, 1)
plt.xlabel("degree")
plt.ylabel("score")

# %%
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)

# %%
X2, y2 = make_data(200)
plt.scatter(X2.ravel(), y2)

# %%
degree = np.arange(21)
train_score2, val_score2 = validation_curve(
    PolynomialRegression(), X2, y2, "polynomialfeatures__degree", degree, cv=7
)

plt.plot(degree, np.median(train_score2, 1), color="blue", label="training score")
plt.plot(degree, np.median(val_score2, 1), color="red", label="validation score")
plt.plot(degree, np.median(train_score, 1), color="blue", alpha=0.3, linestyle="dashed")
plt.legend(loc="lower center")
plt.ylim(0, 1)
plt.xlabel("degree")
plt.ylabel("score")

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(
        PolynomialRegression(degree), X, y, cv=7, train_sizes=np.linspace(0.3, 1, 25)
    )

ax[i].plot(N, np.mean(train_lc, 1), color="blue", label="training score")
ax[i].plot(N, np.mean(val_lc, 1), color="red", label="validation score")
ax[i].hlines(
    np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color="gray", linestyle="dashed"
)

ax[i].set_ylim(0, 1)
ax[i].set_xlim(N[0], N[-1])
ax[i].set_xlabel("training size")
ax[i].set_ylabel("score")
ax[i].set_title("degree = {0}".format(degree), size=14)
ax[i].legend(loc="best")

# %%
param_grid = {
    "polynomialfeatures__degree": np.arange(21),
    "linearregression__fit_intercept": [True, False],
    "linearregression__normalize": [True, False],
}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)

# %%
grid.fit(X, y)

# %%
grid.best_params_

# %%
model = grid.best_estimator_

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)

# %%
data = [
    {"price": 950000, "rooms": 4, "neighborhood": "Queen Anne"},
    {"price": 700000, "rooms": 3, "neighborhood": "Fremont"},
    {"price": 650000, "rooms": 3, "neighborhood": "Wallingford"},
    {"price": 600000, "rooms": 2, "neighborhood": "Fremont"},
]

# %%
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)

# %%
vec.get_feature_names()

# %%
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)

# %%
sample = ["problem of evil", "evil queen", "horizon problem"]

# %%
vec = CountVectorizer()
X = vec.fit_transform(sample)
X


# %%
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# %%
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# %%
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)

# %%
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)

# %%
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)

# %%
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)

# %%
X = np.array([[nan, 0, 3], [3, 7, 9], [3, 5, 2], [4, nan, 6], [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])

# %%
imp = SimpleImputer(strategy="mean")
X2 = imp.fit_transform(X)
X2

# %%
model = LinearRegression().fit(X2, y)
model.predict(X2)

# %%
model = make_pipeline(
    SimpleImputer(strategy="mean"), PolynomialFeatures(degree=2), LinearRegression()
)


# %%
model.fit(X, y)
print(y)
print(model.predict(X))

# %%
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="RdBu")

# %%
model = GaussianNB()
model.fit(X, y)

# %%
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

# %%
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="RdBu")
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap="RdBu", alpha=0.1)
plt.axis(lim)

# %%
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)

# %%
data = fetch_20newsgroups()
data.target_names

# %%
categories = [
    "talk.religion.misc",
    "soc.religion.christian",
    "sci.space",
    "comp.graphics",
]
train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)

# %%
print(train.data[5])

# %%
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# %%
model.fit(train.data, train.target)
labels = model.predict(test.data)

# %%
mat = confusion_matrix(test.target, labels)
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt="d",
    cbar=False,
    xticklabels=train.target_names,
    yticklabels=train.target_names,
)
plt.xlabel("true label")
plt.ylabel("predicted label")

# %%


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# %%
predict_category("sending a payload to the ISS")

# %%
predict_category("discussing islam vs atheism")

# %%
predict_category("determining the screen resolution")

# %%
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y)


# %%
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)

# %%
print("Model slope:", model.coef_[0])
print("Model intercept:", model.intercept_)

# %%
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2.0, 1.0])

model.fit(X, y)
print(model.intercept_)
print(model.coef_)

# %%
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])


# %%
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

# %%
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.rand(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)

# %%


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for 1d input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(
            X[:, :, np.newaxis], self.centers_, self.width_, axis=1
        )


gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10)

# %%
model = make_pipeline(GaussianFeatures(30), LinearRegression())
model.fit(x[:, np.newaxis], y)

plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))

plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)

# %%


def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel="x", ylabel="y", ylim=(-1.5, 1.5))

    if title:
        ax[0].set_title(title)

    ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
    ax[1].set(xlabel="basis location", ylabel="coefficent", xlim=(0, 10))


model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)

# %%
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title="Ridge Regression")

# %%
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title="Lasso Regression")

# %%
# Skip time series example due to data not being accessible

# %%
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")


# %%
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
plt.plot([0.6], [2.1], "x", color="red", markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, "-k")

plt.xlim(-1, 3.5)

# %%
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, "-k")
    plt.fill_between(
        xfit, yfit - d, yfit + d, edgecolor="none", color="#AAAAAA", alpha=0.4
    )

plt.xlim(-1, 3.5)

# %%
model = SVC(kernel="linear", C=1e10)
model.fit(X, y)

# %%


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a two-dimensional SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(
        X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )

    if plot_support:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=300,
            linewidth=1,
            facecolors="none",
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# %%
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
plot_svc_decision_function(model)

# %%
model.support_vectors_

# %%


def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.6)

    X = X[:N]
    y = y[:N]
    model = SVC(kernel="linear", C=1e10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title("N={0}".format(N))

# %%
X, y = make_circles(100, factor=0.1, noise=0.1)

clf = SVC(kernel="linear").fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
plot_svc_decision_function(clf, plot_support=False)

# %%
r = np.exp(-(X ** 2).sum(1))

# %%


def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap="autumn")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


plot_3D()
# %%
clf = SVC(kernel="rbf", C=1e6)
clf.fit(X, y)

# %%
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
plot_svc_decision_function(clf)
plt.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=300,
    lw=1,
    facecolors="none",
)

# %%
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")

# %%
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel="linear", C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="autumn")
    plot_svc_decision_function(model, axi)
    axi.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=300,
        lw=1,
        facecolors="none",
    )
    axi.set_title("C={0:.1f}".format(C), size=14)

# %%
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# %%
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap="bone")
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])

# %%
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel="rbf")
model = make_pipeline(pca, svc)

# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(
    faces.data, faces.target, random_state=42
)

# %%
param_grid = {"svc__C": [1, 5, 10, 50], "svc__gamma": [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

grid.fit(Xtrain, ytrain)
print(grid.best_params_)

# %%
model = grid.best_estimator_
yfit = model.predict(Xtest)


# %%
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap="bone")
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(
        faces.target_names[yfit[i]].split()[-1],
        color="black" if yfit[i] == ytest[i] else "red",
    )

fig.suptitle("Predicted Names; Incorrect Labels in Red", size=14)
# %%
print(classification_report(ytest, yfit, target_names=faces.target_names))

# %%
mat = confusion_matrix(ytest, yfit)
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt="d",
    cbar=False,
    xticklabels=faces.target_names,
    yticklabels=faces.target_names,
)
plt.xlabel("true label")
plt.ylabel("predicted label")

# %%
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")

# %%
tree = DecisionTreeClassifier().fit(X, y)

# %%


def visualize_classifier(model, X, y, ax=None, cmap="rainbow"):
    ax = ax or plt.gca()

    ax.scatter(
        X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3
    )

    ax.axis("tight")
    ax.axis("off")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    contours = ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        levels=np.arange(n_classes + 1) - 0.5,
        cmap=cmap,
        clim=(y.min(), y.max()),
        zorder=1,
    )

    ax.set(xlim=xlim, ylim=ylim)


# %%
visualize_classifier(DecisionTreeClassifier(), X, y)

# %%
tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1)

bag.fit(X, y)
visualize_classifier(bag, X, y)

# %%
model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y)

# %%
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)


def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))

    return slow_oscillation + fast_oscillation + noise


y = model(x)
plt.errorbar(x, y, 0.3, fmt="o")

# %%
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)

xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)

plt.errorbar(x, y, 0.3, fmt="o", alpha=0.5)
plt.plot(xfit, yfit, "-r")
plt.plot(xfit, ytrue, "-k", alpha=0.5)

# %%
digits = load_digits()
digits.keys()

# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(
    digits.data, digits.target, random_state=0
)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

# %%
print(classification_report(ypred, ytest))

# %%
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("true label")
plt.ylabel("predicted label")

# %%
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis("equal")

# %%
pca = PCA(n_components=2)
pca.fit(X)

# %%
print(pca.components_)

# %%
print(pca.explained_variance_)

# %%


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle="->", linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate("", v1, v0, arrowprops=arrowprops)


plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis("equal")

# %%
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape: ", X.shape)
print("transformed shape: ", X_pca.shape)

# %%
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis("equal")


# %%
pca = PCA(2)
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

# %%
plt.scatter(
    projected[:, 0], projected[:, 1], c=digits.target, edgecolors="none", alpha=0.5
)
plt.xlabel("component 1")
plt.ylabel("component 2")

# %%
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")

# %%


def plot_digits(data):
    fig, axes = plt.subplots(
        4,
        10,
        figsize=(10, 4),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            data[i].reshape(8, 8), cmap="binary", interpolation="nearest", clim=(0, 16)
        )


plot_digits(digits.data)

# %%
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)

# %%
pca = PCA(0.5).fit(noisy)
pca.n_components_

# %%
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)

# %%
faces = fetch_lfw_people(min_faces_per_person=60)
pca = PCA(150, svd_solver="randomized")
pca.fit(faces.data)

# %%
fig, axes = plt.subplots(
    3,
    8,
    figsize=(9, 4),
    subplot_kw={"xticks": [], "yticks": []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1),
)
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap="bone")

# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")

# %%
pca = PCA(150, svd_solver="randomized").fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)


# %%
fig, ax = plt.subplots(
    2,
    10,
    figsize=(10, 2.5),
    subplot_kw={"xticks": [], "yticks": []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1),
)
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap="binary_r")
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap="binary_r")

ax[0, 0].set_ylabel("full-dim\ninput")
ax[1, 0].set_ylabel("150-dim\nreconstruction")

# %%


def make_hello(N=1000, rseed=42):
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis = "off"
    ax.text(0.5, 0.4, "HELLO", va="center", ha="center", weight="bold", size=85)
    fig.savefig("hello.png")
    plt.close(fig)

    data = imread("hello.png")[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = data[i, j] < 1
    X = X[mask]
    X[:, 0] *= data.shape[0] / data.shape[1]
    X = X[:N]
    return X[np.argsort(X[:, 0])]


# %%
X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap("rainbow", 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis("equal")

# %%


def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)


X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis("equal")

# %%
D = pairwise_distances(X)
D.shape

# %%
plt.imshow(D, zorder=2, cmap="Blues", interpolation="nearest")
plt.colorbar()

# %%
D2 = pairwise_distances(X2)
np.allclose(D, D2)

# %%
model = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis("equal")

# %%


def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[: X.shape[1]])


X3 = random_projection(X, 3)
X3.shape

# %%
ax = plt.axes(projection="3d")
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2], **colorize)
ax.view_init(azim=70, elev=50)

# %%
model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis("equal")

# %%


def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T


XS = make_hello_s_curve(X)

# %%
ax = plt.axes(projection="3d")
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)

# %%
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis("equal")

# %%
model = LocallyLinearEmbedding(
    n_neighbors=100, n_components=2, method="modified", eigen_solver="dense"
)
out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15)

# %%
faces = fetch_lfw_people(min_faces_per_person=30)
faces.data.shape

# %%
fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap="gray")

# %%
model = PCA(100, svd_solver="randomized").fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel("n components")
plt.ylabel("cumulative variance")

# %%
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)
proj.shape

# %%


def plot_components(data, model, images=None, ax=None, thumb_frac=0.5, cmap="gray"):
    ax = ax or plt.gca()

    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], ".k")

    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap), proj[i]
            )
            ax.add_artist(imagebox)


# %%
fig, ax = plt.subplots(figsize=(10, 10))
plot_components(
    faces.data, model=Isomap(n_components=2), images=faces.images[:, ::2, ::2]
)

# %%
mnist = fetch_openml("mnist_784")
mnist.data.shape

# %%
fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(mnist.data[1250 * i].reshape(28, 28), cmap="gray_r")

# %%
data = mnist.data[::30]
target = mnist.target[::30]

model = Isomap(n_components=2)
proj = model.fit_transform(data)
plt.scatter(proj[:, 0], proj[:, 1])

# %%
data = mnist.data[mnist.target == 1][::4]

fig, ax = plt.subplots(figsize=(10, 10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver="dense")
plot_components(
    data,
    model,
    images=data.reshape((-1, 28, 28)),
    ax=ax,
    thumb_frac=0.05,
    cmap="gray_r",
)

# %%
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)

# %%
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# %%
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)

# %%


def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        labels = pairwise_distances_argmin(X, centers)

        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

# %%
centers, labels = find_clusters(X, 4, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

# %%
labels = KMeans(6, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

# %%
X, y = make_moons(200, noise=0.05, random_state=0)

# %%
labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

# %%
model = SpectralClustering(
    n_clusters=2, affinity="nearest_neighbors", assign_labels="kmeans"
)
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")

# %%
digits = load_digits()
digits.data.shape

# %%
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

# %%
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation="nearest", cmap=plt.cm.binary)

# %%
labels = np.zeros_like(clusters)
for i in range(10):
    mask = clusters == i
    labels[mask] = mode(digits.target[mask])[0]

# %%
accuracy_score(digits.target, labels)

# %%
mat = confusion_matrix(digits.target, labels)
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt="d",
    cbar=False,
    xticklabels=digits.target_names,
    yticklabels=digits.target_names,
)
plt.xlabel("true label")
plt.ylabel("predicted label")

# %%
tsne = TSNE(n_components=2, init="pca", random_state=0)
digits_proj = tsne.fit_transform(digits.data)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

labels = np.zeros_like(clusters)
for i in range(10):
    mask = clusters == i
    labels[mask] = mode(digits.target[mask])[0]

accuracy_score(digits.target, labels)

# %%
china = load_sample_image("china.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)

# %%
china.shape

# %%
data = china / 255.0
data = data.reshape(427 * 640, 3)
data.shape

# %%


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker=".")
    ax[0].set(xlabel="Red", ylabel="Green", xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker=".")
    ax[1].set(xlabel="Red", ylabel="Blue", xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)


# %%
plot_pixels(data, title="Input color space: 16 million possible colors")

# %%
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title="Reduced color space: 16 colors")

# %%
china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title("Original Image", size=16)
ax[1].imshow(china_recolored)
ax[1].set_title("16-color Image", size=16)

# %%
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
X = X[:, ::-1]

# %%
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis")

# %%


def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    ax = ax or plt.gca()
    ax.axis("equal")
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis", zorder=2)

    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc="#CCCCCC", lw=3, alpha=0.5, zorder=1))


# %%
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

# %%
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

# %%
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis")

# %%
probs = gmm.predict_proba(X)
print(probs[:5].round(3))

# %%
size = 50 * probs.max(1) ** 2
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=size)

# %%


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis", zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, zorder=2)
    ax.axis("equal")

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


# %%
gmm = GaussianMixture(n_components=4, random_state=42)
plot_gmm(gmm, X)

# %%
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
plot_gmm(gmm, X_stretched)

# %%
Xmoon, ymoon = make_moons(200, noise=0.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1])

# %%
gmm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
plot_gmm(gmm2, Xmoon)

# %%
gmm16 = GaussianMixture(n_components=16, covariance_type="full", random_state=0)
plot_gmm(gmm16, Xmoon, label=False)

# %%
# Xnew = gmm16.sample(400)
# plt.scatter(Xnew[:, 0], Xnew[:, 1])

# %%
n_components = np.arange(1, 21)
models = [
    GaussianMixture(n, covariance_type="full", random_state=0).fit(Xmoon)
    for n in n_components
]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label="BIC")
plt.plot(n_components, [m.aic(Xmoon) for m in models], label="AIC")
plt.legend(loc="best")
plt.xlabel("n_components")

# %%
digits = load_digits()
digits.data.shape

# %%


def plot_digits(data):
    fig, ax = plt.subplots(
        10, 10, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[])
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap="binary")
        im.set_clim(0, 16)
    plot_digits(digits.data)


# %%
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
data.shape

# %%
n_components = np.arange(50, 210, 10)
models = [
    GaussianMixture(n, covariance_type="full", random_state=0) for n in n_components
]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics)

# %%
gmm = GaussianMixture(110, covariance_type="full", random_state=0)
gmm.fit(data)
print(gmm.converged_)

# %%
data_new = gmm.sample(n_samples=100)
data_new[0].shape

# %%
digit_new = pca.inverse_transform(data_new[0])
plot_digits(digit_new)

# %%


def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x


x = make_data(1000)

# %%
hist = plt.hist(x, bins=30, normed=True)

# %%
density, bins, patches = hist
widths = bins[1:] - bins[:-1]
(density * widths).sum()

# %%
x = make_data(20)
bins = np.linspace(-5, 10, 10)

# %%
fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 4),
    sharex=True,
    sharey=True,
    subplot_kw={"xlim": (-4, 9), "ylim": (-0.02, 0.3)},
)
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, normed=True)
    ax[i].plot(x, np.full_like(x, -0.01), "|k", markeredgewidth=1)

# %%
fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), "|k", markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1, alpha=0.5))
ax.set_xlim(-4, 8)
ax.set_ylim(-0.2, 8)

# %%
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), "|k", markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8])

# %%
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), "|k", markeredgewidth=1)

plt.axis([-4, 8, -0.2, 5])


# %%
kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
kde.fit(x[:, None])

logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), "|k", markeredgewidth=1)
plt.ylim(-0.02, 0.22)

# %%
bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(
    KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=LeaveOneOut()
)
grid.fit(x[:, None])

# %%
grid.best_params_

# %%
# skip geomapping example- not high value & has installation requirements

# %%


class KDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [
            KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi)
            for Xi in training_sets
        ]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# %%
digits = load_digits()

bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {"bandwidth": bandwidths})
grid.fit(digits.data, digits.target)

# %%
# scores = [val.mean_validation_score for val in grid.cv_results_]

# %%
# plt.semilogx(bandwidths, scores)
# plt.xlabel('bandwidth')
# plt.ylabel('accuracy')
# plt.title('KDE Model Performance')
print(grid.best_params_)
print("accuracy=", grid.best_score_)

# %%
cross_val_score(GaussianNB(), digits.data, digits.target).mean()

# %%
image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualise=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap="gray")
ax[0].set_title("input image")

ax[1].imshow(hog_vis)
ax[1].set_title("visualization of HOG features")

# %%
faces = fetch_lfw_people()
positive_patches = faces.images
postitive_patches.shape

# %%
imgs_to_use = [
    "camera",
    "text",
    "coins",
    "moon",
    "page",
    "clock",
    "immunohistochemistry",
    "chelsea",
    "coffee",
    "hubble_deep_field",
]
images = [color.rgb2gray(getattr(data, name)) for name in imgs_to_use]

# %%


def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(
        patch_size=extracted_patch_size, max_patches=N, random_state=0
    )
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transfom.resize(patch, patch_size) for patch in patches])

    return patches


negative_patches = np.vstack(
    [extract_patches(im, 1000, scale) for im in images for scale in [0.5, 1.0, 2.0]]
)

negative_patches.shape

# %%
fig, ax = plt.subplots(6, 10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500 * i], cmap="gray")
    axi.axis("off")

# %%
X_train = np.array(
    [feature.hog(im) for im in chain(positive_patches, negative_patches)]
)
y_train = np.zeroes(X_train.shape[0])
y_train[: positive_patches.shape[0]] = 1

# %%
X_train.shape

# %%
cross_val_score(GaussianNB(), X_train, ytrain)

# %%
grid = GridSearchCV(LinearSVC(), {"C": [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
grid.best_score_

# %%
grid.best_params_

# %%
model = grid.best_estimator_
model.fit(X_train, y_train)

# %%
test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = skimage.transform.rescale(test_image, 0.5)
test_image = test_image[:160, 40:180]

plt.imshow(test_image, cmap="gray")
plt.axis("off")

# %%


def sliding_window(
    img, patch_size=positive_patches[0].shape, istep=2, jstep=2, scale=1.0
):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i : i + Ni, j : j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([feature.hog(patch) for patch in patches])
patches_hog.shape

# %%
labels = model.predict(patches_hog)
labels.sum()

# %%
fig, ax = plt.subplots()
ax.imshow(test_image, cmap="gray")
ax.axis("off")

Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(
        plt.Rectangle(
            (j, i), Nj, Ni, edgecolor="red", alpha=0.3, lw=2, facecolor="none"
        )
    )
