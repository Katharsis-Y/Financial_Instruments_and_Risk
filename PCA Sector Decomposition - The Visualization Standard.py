import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# ---------------------------------------------------------
# [Protocol] "Quant Research" Publication Style (Final Polish)
# ---------------------------------------------------------
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8

# Palette
COLOR_BG = '#FFFFFF'
COLOR_GRID = '#E0E0E0'
COLOR_TEXT = '#2C3E50'
COLOR_BAR = '#2980b9'
COLOR_LINE = '#c0392b'
COLOR_PC1 = '#004C6D'
COLOR_MARKET = '#95a5a6'

plt.rcParams['figure.facecolor'] = COLOR_BG
plt.rcParams['axes.facecolor'] = COLOR_BG
plt.rcParams['text.color'] = COLOR_TEXT
plt.rcParams['axes.labelcolor'] = COLOR_TEXT
plt.rcParams['xtick.color'] = COLOR_TEXT
plt.rcParams['ytick.color'] = COLOR_TEXT
plt.rcParams['grid.color'] = COLOR_GRID

class InstitutionalPCA:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        self.raw_data = None
        self.returns = None
        self.pca_model = None
        self.pca_scores = None
        self.loadings = None
        self.scaler = StandardScaler()
        
    def fetch_and_process(self):
        print(f"[*] Fetching Market Data...")
        df = yf.download(self.tickers, start=self.start, end=self.end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df['Adj Close']
            except KeyError: df = df['Close']
        self.raw_data = df.dropna()
        self.returns = np.log(self.raw_data / self.raw_data.shift(1)).dropna()
        self.scaled_returns = pd.DataFrame(self.scaler.fit_transform(self.returns), 
                                         columns=self.returns.columns, index=self.returns.index)

    def run_engine(self):
        self.pca_model = PCA(n_components=None)
        self.pca_scores = self.pca_model.fit_transform(self.scaled_returns)
        n_cols = self.pca_model.n_components_
        self.loadings = pd.DataFrame(self.pca_model.components_.T, 
                                   columns=[f'PC{i+1}' for i in range(n_cols)], index=self.tickers)
        if self.loadings['PC1'].mean() < 0:
            self.loadings['PC1'] = -1 * self.loadings['PC1']
            self.pca_scores[:, 0] = -1 * self.pca_scores[:, 0]

    def generate_report(self):
        fig = plt.figure(figsize=(16, 12))
        
        # Header
        fig.suptitle('Principal Component Analysis: S&P 500 Sector Decomposition (2021-2023)', 
                     fontsize=18, fontweight='bold', y=0.97, color='#2c3e50')
        
        # Grid Layout
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25, 
                             left=0.08, right=0.92, top=0.90, bottom=0.08)
        
        # -----------------------------------------------------
        # Chart A: Scree Plot
        # -----------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        var_ratio = self.pca_model.explained_variance_ratio_
        cum_var = np.cumsum(var_ratio)
        
        bars = ax1.bar(range(1, len(var_ratio)+1), var_ratio, alpha=0.85, color=COLOR_BAR, zorder=3)
        ax1.set_ylabel('Explained Variance', fontweight='bold')
        ax1.set_xlabel('Principal Component')
        ax1.set_title('A. Dimensionality Profile (Scree Plot)', loc='left', fontsize=12, fontweight='bold', pad=10)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.grid(True, axis='y', zorder=0)
        
        ax1b = ax1.twinx()
        ax1b.plot(range(1, len(var_ratio)+1), cum_var, color=COLOR_LINE, marker='o', markersize=5, linewidth=2)
        ax1b.set_ylabel('Cumulative %', color=COLOR_LINE, fontweight='bold')
        ax1b.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1b.grid(False)
        ax1b.axhline(0.90, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.7)
        
        # [FIX 1] Move label UP slightly (y=0.915) and center it horizontally over the last few bars
        # Also added a white background box to ensure the line doesn't cut through the text
        ax1b.text(len(var_ratio)-2, 0.915, '90% Threshold', va='bottom', ha='center', 
                  fontsize=9, color='#7f8c8d', fontstyle='italic', 
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # -----------------------------------------------------
        # Chart B: Factor Loadings
        # -----------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(self.loadings.iloc[:, :5], annot=True, cmap='RdBu_r', center=0, ax=ax2, fmt='.2f', 
                    cbar_kws={'label': 'Loading Strength', 'shrink': 0.8}, 
                    annot_kws={"size": 9}, linewidths=1, linecolor='white', square=False)
        ax2.set_title('B. Factor Logic (Loadings)', loc='left', fontsize=12, fontweight='bold', pad=10)
        ax2.set_ylabel('Sector Assets')

        # -----------------------------------------------------
        # Chart C: Orthogonality
        # -----------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0])
        scores_subset = pd.DataFrame(self.pca_scores[:, :5], columns=[f'PC{i+1}' for i in range(5)])
        corr_matrix = scores_subset.corr()
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask, k=1)] = True 
        
        sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='Blues', vmin=0, vmax=1, ax=ax3, fmt='.2f', cbar=False,
                    linewidths=2, linecolor='white', square=True, 
                    annot_kws={"size": 10, "weight": "bold", "color": "#2c3e50"}) 
        ax3.set_title('C. Orthogonality Verification', loc='left', fontsize=12, fontweight='bold', pad=10)

        # -----------------------------------------------------
        # Chart D: Time Series (Validation)
        # -----------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1])
        market_idx = (1 + self.returns.mean(axis=1)).cumprod()
        market_idx = 100 * market_idx / market_idx.iloc[0]
        pc1_level = pd.Series(np.cumsum(self.pca_scores[:, 0]), index=self.returns.index)
        
        l1, = ax4.plot(market_idx, color=COLOR_MARKET, linewidth=1.5, linestyle='-', label='S&P Equal Weight (Left)')
        ax4.set_ylabel('Market Index (100 Base)', color=COLOR_MARKET, fontweight='bold')
        
        ax4b = ax4.twinx()
        l2, = ax4b.plot(pc1_level, color=COLOR_PC1, linewidth=2.5, label='PC1 Factor Score (Right)')
        ax4b.set_ylabel('Factor Score Level', color=COLOR_PC1, fontweight='bold')
        ax4b.grid(False)
        
        ax4.set_title('D. Validation: PC1 vs Market Regime', loc='left', fontsize=12, fontweight='bold', pad=15)
        
        # [FIX 2] Move Legend INSIDE the plot (Top Center) with transparency
        # y=0.96 places it just below the top axis spine.
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                   ncol=2, frameon=True, framealpha=0.9, fontsize=10, edgecolor='#ccc')

        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha='center')

        fig.text(0.08, 0.02, 'Source: Yahoo Finance | Methodology: PCA Decomposition Strategy', 
                 fontsize=9, color='#7f8c8d', style='italic')

        plt.savefig('pca_publication_final_v4.png', bbox_inches='tight')
        print("[System] Final Refined Report generated: pca_publication_final_v4.png")
        plt.show()

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    tickers = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
    model = InstitutionalPCA(tickers, '2021-01-01', '2023-12-31')
    model.fetch_and_process()
    model.run_engine()
    model.generate_report()