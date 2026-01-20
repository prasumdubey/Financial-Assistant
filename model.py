
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data.csv")

# Feature Engineering
income = df.iloc[:, [37, 38, 44, 45, 48, 49, 55, 66, 73, 78, 79, 83, 87, 90]].mean(axis=1)
assets = df.iloc[:, [1, 2, 3, 16, 51, 53, 54, 74, 84]].mean(axis=1)
expenses = df.iloc[:, [9, 11, 12, 15, 16, 17, 34, 43, 52, 59, 60, 63, 68]].mean(axis=1)
debts = df.iloc[:, [10, 14, 35, 36, 39, 89]].mean(axis=1)
liabilities = df.iloc[:, [40, 56, 58, 61, 62, 64, 65, 75, 76, 82, 88]].mean(axis=1)
savings = df.iloc[:, [13, 31, 66, 72, 85, 92]].mean(axis=1)
profit = df.iloc[:, [4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 41, 42, 50, 86]].mean(axis=1)

# Final dataset
data = pd.DataFrame({
    'income': income,
    'assets': assets,
    'expenses': expenses,
    'debts': debts,
    'liabilities': liabilities,
    'savings': savings,
    'profit': profit,
    'can_invest': df['Bankrupt?']
})

X = data.drop('can_invest', axis=1)
y = data['can_invest']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_resampled, y_train_resampled)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save model
pickle.dump(model, open("investment_predictor.pkl", "wb"))

print("Model saved as investment_predictor.pkl")

