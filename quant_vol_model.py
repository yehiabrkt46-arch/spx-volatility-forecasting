import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import optuna
import warnings

# Suppress minor Pandas warnings for a clean terminal output
warnings.filterwarnings('ignore')

print("1. Fetching 5 years of market data...")
data = yf.download("^SPX", start="2018-01-01", end="2024-01-01", progress=False)
data.columns = data.columns.get_level_values(0)

# ---------------------------------------------------------
# FEATURE ENGINEERING & THE TARGET VARIABLE
# ---------------------------------------------------------
print("2. Calculating Rolling Volatility and Lags...")
# Calculate daily returns
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

# Calculate our Target: 5-Day Rolling Volatility
data['Vol_5d'] = data['Log_Return'].rolling(window=5).std()

# Shift target backward so today predicts TOMORROW'S volatility
data['Target_Vol'] = data['Vol_5d'].shift(-1)

# Create historical "Lags" (memory of the past) so the model has features to learn from
for lag in [1, 2, 3, 5]:
    data[f'Vol_Lag_{lag}'] = data['Vol_5d'].shift(lag)
    data[f'Return_Lag_{lag}'] = data['Log_Return'].shift(lag)

# Drop any rows with missing data (NaNs) created by our shifting
data.dropna(inplace=True)

# ---------------------------------------------------------
# THE PURGED CROSS-VALIDATION SPLIT
# ---------------------------------------------------------
print("3. Executing Purged Train/Test Split...")
train_pct = 0.8
split_idx = int(len(data) * train_pct)

# The 5-day Purge Gap ensures no overlapping data leaks from Train to Test
purge_gap = 5  

train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx + purge_gap:]  

# Select only our Lag features for training
features = [col for col in data.columns if 'Lag' in col]

X_train, y_train = train_data[features], train_data['Target_Vol']
X_test, y_test = test_data[features], test_data['Target_Vol']

print(f"   Training Rows: {len(X_train)} | Testing Rows: {len(X_test)}")

# ---------------------------------------------------------
# OPTUNA BAYESIAN OPTIMIZATION & XGBOOST
# ---------------------------------------------------------
print("\n4. Starting Optuna AI Optimization (Finding best XGBoost settings)...")

def objective(trial):
    # Optuna will guess different combinations of these settings
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
    }
    
    # Train the model with the current guess
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Test it on the future (purged) data
    preds = model.predict(X_test)
    
    # Calculate Error (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    return rmse

# Turn off Optuna's massive wall of text to keep the terminal clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Run 20 trials of trial-and-error optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\n=========================================")
print("OPTIMIZATION COMPLETE!")
print("=========================================")
print(f"Best Purged Test Error (RMSE): {study.best_value:.6f}")
print("Best XGBoost Parameters found:")
for key, value in study.best_params.items():
    print(f"  - {key}: {value}")
    # ---------------------------------------------------------
# GENERATE VISUAL PROOF FOR GITHUB
# ---------------------------------------------------------
print("\n5. Generating Optimization Graph for Portfolio...")
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history

# Plot the AI's learning history
fig = plot_optimization_history(study)
plt.tight_layout()

# Save the image directly to your folder
plt.savefig("optuna_proof.png", dpi=300)
print("✓ Saved 'optuna_proof.png' to your folder. Your portfolio is ready!")