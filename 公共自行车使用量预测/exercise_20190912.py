# %% 导入包
import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import mutual_info_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import minmax_scale, FunctionTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load
import tqdm


# %% 对指定列one_hot
def one_hot(df):
    result = pd.get_dummies(df, columns=["weather"])
    return result


# %% 测试模型结果
def evaluate(estimator, X_train, X_test, y_train, y_test):
    estimator.fit(X_train, y_train)
    result = mean_squared_error(y_test, estimator.predict(X_test))
    return result


# %% 读取数据
data = pd.read_csv("data/train.csv", index_col="id")
data.info()
X = data.drop("y", axis=1)
y = data["y"].copy()
print(data.isnull().sum())

# %% 进行特征选择
pearsonr_corr = np.eye(data.shape[1])
mutual_info = np.zeros((data.shape[1], data.shape[1]))
for i in tqdm.tqdm(range(data.shape[1])):
    for j in range(i+1, data.shape[1]):
        pearsonr_corr[i, j] = pearsonr(data.iloc[:, i].values, data.iloc[:, j].values)[0]
        pearsonr_corr[j, i] = pearsonr_corr[i, j]
        mutual_info[i, j] = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
        mutual_info[j, i] = mutual_info[i, j]
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(pearsonr_corr)
plt.subplot(1, 2, 2)
plt.imshow(mutual_info)
plt.colorbar()
plt.show()

# %% 探索每个特征对于自行车使用情况的影响，并可视化
select = RFECV(
    estimator=LogisticRegression(n_jobs=-1), cv=5, verbose=1,
    scoring="accuracy", n_jobs=-1, min_features_to_select=5
)
select.fit(X, y)
print(select.support_)
print(select.n_features_)
print(select.ranking_)

# %% 分别以简单线性回归模型，决策树回归模型等回归模型进行拟合并进行检验，分析检验结果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
X_train = minmax_scale(X_train)
X_test = minmax_scale(X_test)
enc = SelectKBest(mutual_info_regression, k=5).fit(X_train, y_train)
X_train, X_test = enc.transform(X_train), enc.transform(X_test)
mse = []
for model in tqdm.tqdm([
    LinearRegression(n_jobs=-1), LogisticRegression(n_jobs=-1), SVR(),
    DecisionTreeRegressor(), RandomForestRegressor(n_estimators=30, n_jobs=-1), MLPRegressor(hidden_layer_sizes=12)
]):
    mse.append(evaluate(model, X_train, X_test, y_train, y_test))
plt.figure(2)
plt.plot(range(len(mse)), mse)
plt.show()

# %% 使用随机森林模型进行预测
data_train = pd.read_csv("data/train.csv", index_col="id")
X_train = data_train.drop("y", axis=1)
y_train = data_train["y"].copy()
data_test = pd.read_csv("data/test.csv", index_col="id")
params = {
    "n_estimators": range(3, 40),
    "max_depth": range(2, 4)
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(n_jobs=-1, random_state=1), n_jobs=-1,
    verbose=1, param_grid=params, scoring="neg_mean_squared_error", iid=False, cv=5
)
pipeline = Pipeline([
    ("one_hot", FunctionTransformer(one_hot, validate=False)),
    ("minmax_scale", MinMaxScaler())
])
X_train = pipeline.fit_transform(X_train)
try:
    model = load("model/RandomForestRegressor.pkl")
except FileNotFoundError:
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    model = grid_search.best_estimator_
    dump(model, "model/RandomForestRegressor.pkl")
print(mean_squared_error(y_train, model.predict(X_train)))
X_test = pipeline.fit_transform(data_test)
y_test = model.predict(X_test)

submit = data_test.copy()
submit["y"] = y_test
submit.to_csv("data/submit.csv", columns=["y"], index=True)



