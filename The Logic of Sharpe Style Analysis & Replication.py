import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# ---------------------------------------------------------
# 1. THE ARENA (Data Setup)
# ---------------------------------------------------------
TARGET = 'ARKK' 

FACTORS = ['QQQ', 'IWM', 'VLUE', 'AGG', 'GLD'] 
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'

print(f"mary_protocol >> Fetching Battle Data: {TARGET} vs. The Market...")
tickers = [TARGET] + FACTORS
data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
returns = data.pct_change().dropna()

y = returns[TARGET]
X = returns[FACTORS]

# ---------------------------------------------------------
# 2. THE FIVE CONTENDERS (Training Models)
# ---------------------------------------------------------
# Container for weights
weights_df = pd.DataFrame(index=FACTORS)
models = {}

# --- A. OLS (The Glutton) ---
ols = LinearRegression(fit_intercept=False) # No alpha, purely beta
ols.fit(X, y)
weights_df['OLS (Unconstrained)'] = ols.coef_
models['OLS'] = ols

# --- B. Constrained (The Mutual Fund Manager) ---
# Must sum to 1, No Shorts
def objective(w, X, y): return np.sum((y - np.dot(X, w))**2)
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bnds = tuple((0, 1) for _ in range(X.shape[1]))
init_guess = [1/len(FACTORS)] * len(FACTORS)
res = minimize(objective, init_guess, args=(X, y), method='SLSQP', bounds=bnds, constraints=cons)
weights_df['Constrained (Long Only)'] = res.x

# --- Pre-processing for ML ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ML Models output "Feature Importance", need to be scaled back to "Weights" roughly for visualization
# But strictly, we use them to predict first.

# --- C. Lasso (The Minimalist) ---
lasso = LassoCV(cv=5, fit_intercept=False).fit(X_scaled, y)
# To make "Synthetic Returns" for Lasso, we use the scaled model prediction
# We store raw coefs just for the heatmap
weights_df['Lasso (Feature Selection)'] = lasso.coef_ 

# ---------------------------------------------------------
# 3. THE CLONE WARS (Simulation)
# ---------------------------------------------------------
# We reconstruct the "Synthetic" ARKK based on the model's understanding
synthetic_returns = pd.DataFrame(index=y.index)
synthetic_returns['Actual ARKK'] = y

# OLS Clone: dot product of raw weights and factors
synthetic_returns['OLS Clone'] = np.dot(X, weights_df['OLS (Unconstrained)'])

# Constrained Clone: dot product of constrained weights and factors
synthetic_returns['Constrained Clone'] = np.dot(X, weights_df['Constrained (Long Only)'])

# Lasso Clone: Prediction using scaled input (Handles the math correctly)
synthetic_returns['Lasso Clone'] = lasso.predict(X_scaled)

# Calculate Cumulative Returns (The P&L Curve)
cumulative_returns = (1 + synthetic_returns).cumprod()

# ---------------------------------------------------------
# 4. VISUALIZATION (Perception Mode)
# ---------------------------------------------------------
plt.style.use('default')
fig = plt.figure(figsize=(16, 10), facecolor='white')

# --- Plot 1: The Race (Cumulative Returns) ---
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1.set_facecolor('white')
ax1.grid(True, color='#d3d3d3', linestyle='--', alpha=0.5)

# Plot Actual
ax1.plot(cumulative_returns.index, cumulative_returns['Actual ARKK'], 
         color='black', linewidth=3, label='Actual ARKK (The Target)', alpha=0.8)

# Plot Clones
ax1.plot(cumulative_returns.index, cumulative_returns['OLS Clone'], 
         color='#1f77b4', linewidth=2, linestyle='--', label='OLS Clone (Cheater)')
ax1.plot(cumulative_returns.index, cumulative_returns['Constrained Clone'], 
         color='#ff7f0e', linewidth=2, linestyle='-', label='Constrained Clone (Realistic)')
ax1.plot(cumulative_returns.index, cumulative_returns['Lasso Clone'], 
         color='#2ca02c', linewidth=2, linestyle=':', label='Lasso Clone (Sparse)')

ax1.set_title("The Clone Wars: Which Model Can Replicate ARKK?", fontsize=14, fontweight='bold')
ax1.legend()
ax1.set_ylabel("Growth of $1 Investment")

# --- Plot 2: The Recipe (Weights Heatmap) ---
ax2 = plt.subplot2grid((2, 2), (1, 0))
sns.heatmap(weights_df[['OLS (Unconstrained)', 'Constrained (Long Only)']], 
            annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax2, cbar=False)
ax2.set_title("The Weights: OLS vs Constrained", fontsize=12)

# --- Plot 3: Lasso Weights (Separate to handle scale diff) ---
ax3 = plt.subplot2grid((2, 2), (1, 1))
# Plotting Lasso coefs as a bar chart to show "Zeroing Out"
lasso_coefs = pd.Series(lasso.coef_, index=FACTORS)
lasso_coefs.plot(kind='barh', ax=ax3, color='#2ca02c')
ax3.set_title("Lasso's Choice: Feature Importance (Scaled)", fontsize=12)
ax3.grid(True, axis='x', linestyle='--')
ax3.set_xlabel("Coefficient Magnitude (Z-Score space)")

# --- Metric Report ---
ols_lev = np.sum(np.abs(weights_df['OLS (Unconstrained)']))
cons_lev = np.sum(np.abs(weights_df['Constrained (Long Only)']))
print("\n--- MARY PROTOCOL: MODEL AUDIT ---")
print(f"1. OLS Leverage Ratio: {ols_lev:.2f}x (Gambler!)")
print(f"2. Constrained Leverage: {cons_lev:.2f}x (Honest 1.0)")
print(f"3. Lasso Non-Zero Factors: {np.sum(lasso.coef_ != 0)} / {len(FACTORS)} (Picker)")

plt.tight_layout()
plt.show()