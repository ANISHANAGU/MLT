import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# ---------------- FEATURES ----------------
# We use beneficiary_count + encoded month patterns
df["month_num"] = pd.to_datetime(df["month"]).dt.month

X = df[["beneficiary_count", "month_num"]]
y = df["total_foodgrain_distributed_tonnes"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")