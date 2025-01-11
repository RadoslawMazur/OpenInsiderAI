from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

df = pd.read_csv("dataset.csv")
n_weeks = 2

df_train = df[df["abs_week"].between(700, 1100)]
df_test = df[df["abs_week"].between(1101, 1200)]

X_train = df_train[[
    'qty', 'owned',
    'delta_owned', 'value', 'is_dir', 'is_ceo', 'is_major_steakholder',
    'd_to_filling', 'A - Grant', 'C - Converted deriv',
    'D - Sale to issuer', 'F - Tax', 'G - Gift', 'M - OptEx',
    'P - Purchase', 'S - Sale', 'W - Inherited', 'X - OptEx',
    '52w_GDP_change', '104w_GDP_change', 'FEDFUNDS', '26w_FEDFUNDS_change',
    '52w_FEDFUNDS_change', '-1w_change']].values

y_train = df_train[[f"{i}w_change" for i in range(1, n_weeks)]].values


X_test = df_test[[
    'qty', 'owned',
    'delta_owned', 'value', 'is_dir', 'is_ceo', 'is_major_steakholder',
    'd_to_filling', 'A - Grant', 'C - Converted deriv',
    'D - Sale to issuer', 'F - Tax', 'G - Gift', 'M - OptEx',
    'P - Purchase', 'S - Sale', 'W - Inherited', 'X - OptEx',
    '52w_GDP_change', '104w_GDP_change', 'FEDFUNDS', '26w_FEDFUNDS_change',
    '52w_FEDFUNDS_change', '-1w_change']].values

y_test = df_test[[f"{i}w_change" for i in range(1, n_weeks)]].values

model = RandomForestRegressor(n_estimators=10_000, min_samples_split=20, 
                              oob_score=True, max_samples=5_000, verbose=3, n_jobs=-1)

model.fit(X_train, y_train)
print("Model trained")

joblib.dump(model, "rfr.joblib")
print("Model saved")

y_hat_train = model.predict(X_train[:500])
print("Train set MSE")
print(mean_squared_error(y_train[:500], y_hat_train[:500]))

y_hat = model.predict(X_test)
print("Test set MSE")
print(mean_squared_error(y_test, y_hat))
