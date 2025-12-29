import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. DATA COLLECTION [cite: 32, 33]
try:
    df = pd.read_csv('student-mat.csv', sep=';')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'student-mat.csv' not found in this folder.")
    exit()

# 2. FEATURE ENGINEERING (Advanced Step) [cite: 35]
# Creating a 'Total Education' feature from Mother and Father's education [cite: 16]
df['Total_Edu'] = df['Medu'] + df['Fedu']
# Creating a 'Social_Index' to see impact of social life on grades [cite: 16]
df['Social_Index'] = df['goout'] + df['Dalc'] + df['Walc']

# Selecting features based on your objectives [cite: 22, 26]
features = ['studytime', 'failures', 'absences', 'G1', 'G2', 'Total_Edu', 'Social_Index', 'health', 'freetime']
X = df[features]
y = df['G3']

# 3. DATA PREPROCESSING [cite: 34, 35]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL DEVELOPMENT & COMPARISON 
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost/GradientBoost": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

best_model = None
best_r2 = -float('inf')
results = []

print("\n--- Model Performance Comparison ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    results.append({"Model": name, "R2 Score": r2, "MAE": mae})
    
    print(f"{name}: R2 = {r2:.4f}, MAE = {mae:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# 5. SAVE THE BEST PERFORMING MODEL [cite: 28]
with open('student_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n🌟 Best Model: {best_model_name} (Saved as 'student_model.pkl')")

# 6. EXPLORATORY DATA ANALYSIS (Visualization) [cite: 36, 37, 43]
plt.figure(figsize=(12, 6))

# Plot 1: Feature Importance for the Best Model
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=features)
    importances.nlargest(7).plot(kind='barh', color='skyblue')
    plt.title(f"Top Factors Impacting Performance ({best_model_name})")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("📈 Feature importance chart saved as 'feature_importance.png'")
    plt.show()

# Plot 2: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_model.predict(X_test), alpha=0.5, color='green')
plt.plot([0, 20], [0, 20], '--r') # Diagonal line
plt.title("Actual vs Predicted Final Grades")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.savefig('prediction_accuracy.png')
plt.show()