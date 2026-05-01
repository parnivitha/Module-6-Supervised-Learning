import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df_preprocess = pd.read_csv("agrofood_co2_emission.csv")
df_preprocess.columns = df_preprocess.columns.str.strip().str.lower()

print("Rows loaded (before cleaning):", len(df_preprocess))

df_clean = df_preprocess.dropna()

print("Rows after dropping missing data:", len(df_clean))

df_original = df_clean.copy()

df = pd.get_dummies(df_clean, columns=["area"], drop_first=True)

y = df["total_emission"]

X = df.drop(columns=["total_emission"])

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=42
)

train_rmse = []
test_rmse = []
train_r2 = []
test_r2 = []

for depth in range(1, 21):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X_train, y_train)

    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)

    train_rmse.append(rmse_train)
    test_rmse.append(rmse_test)
    train_r2.append(r2_train)
    test_r2.append(r2_test)

    print(
        f"depth = {depth}: "
        f"train_RMSE = {rmse_train:.3f}, "
        f"test_RMSE = {rmse_test:.3f}, "
        f"train_R2 = {r2_train:.3f}, "
        f"test_R2 = {r2_test:.3f}"
    )

plt.plot(range(1, 21), train_rmse, marker="o", label="Train RMSE")
plt.plot(range(1, 21), test_rmse, marker="o", label="Test RMSE")
plt.xlabel("max_depth")
plt.ylabel("RMSE")
plt.title("Decision Tree RMSE vs Depth")
plt.legend()
plt.show()

plt.plot(range(1, 21), train_r2, marker="o", label="Train R2")
plt.plot(range(1, 21), test_r2, marker="o", label="Test R2")
plt.xlabel("max_depth")
plt.ylabel("R2 Score")
plt.title("Decision Tree R2 vs Depth")
plt.legend()
plt.show()

best_depth = 15

final_reg = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
final_reg.fit(X_train, y_train)

pred_test = final_reg.predict(X_test)

results = df_original.loc[df_test.index].copy()

results["actual"] = y_test
results["predicted"] = pred_test
results["absolute_error"] = abs(results["actual"] - results["predicted"])

wrong_samples = results.sort_values(by="absolute_error", ascending=False).head(5)

cols_to_show = [
   "area",
    "year",
    "savanna fires",
    "forest fires",
    "crop residues",
    "rice cultivation",
    "drained organic soils (co2)",
    "pesticides manufacturing",
    "forestland",
    "food transport",
    "food retail",
    "food household consumption",
    "on-farm electricity use",
    "food packaging",
    "agrifood systems waste disposal",
    "food processing",
    "fertilizers manufacturing",
    "ippu",
    "manure management",
    "rural population",
    "urban population",
]

selected = wrong_samples[cols_to_show]

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print("\nTop 5 largest errenous countries and associated features:")
print(selected.T)

print("\nTop 5 largest errors:")
print(wrong_samples[cols_to_show[:2] + ["actual", "predicted", "absolute_error"]])