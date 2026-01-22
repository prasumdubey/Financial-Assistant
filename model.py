import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import randint

df = pd.read_csv("C:/Users/Prashant dubey/Desktop/data.csv")

income = df.iloc[:, [37,38,44,45,48,49,55,66,73,78,79,83,87,90]].mean(axis=1)
assets = df.iloc[:, [1,2,3,16,51,53,54,74,84]].mean(axis=1)
expenses = df.iloc[:, [9,11,12,15,16,17,34,43,52,59,60,63,68]].mean(axis=1)
debts = df.iloc[:, [10,14,35,36,39,89]].mean(axis=1)
liabilities = df.iloc[:, [40,56,58,61,62,64,65,75,76,82,88]].mean(axis=1)
savings = df.iloc[:, [13,31,66,72,85,92]].mean(axis=1)
profit = df.iloc[:, [4,5,6,7,8,18,19,20,21,22,23,24,25,26,27,28,29,30,41,42,50,86]].mean(axis=1)

# Final structured dataset
data = pd.DataFrame({
    'income': income,
    'assets': assets,
    'expenses': expenses,
    'debts': debts,
    'liabilities': liabilities,
    'savings': savings,
    'profit': profit
})

# Target variable (safe investment amount logic)
key = df.iloc[:, 29]
target = df.iloc[:, 32] * key


X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),     # Handle missing values
    ('scaler', StandardScaler()),                    # Feature scaling
    ('model', GradientBoostingRegressor(random_state=42))  # Model
])

param_dist = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': randint(3, 7),
    'model__min_samples_split': randint(2, 10),
    'model__min_samples_leaf': randint(1, 4),
    'model__subsample': [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)



# Save model
pickle.dump(model, open("investment_predictor.pkl", "wb"))

print("Model saved as investment_predictor.pkl")

