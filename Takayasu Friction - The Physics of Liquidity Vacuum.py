import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats

# --- Configuration & Styling ---
# 使用 Seaborn 的论文风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif' # 保证跨平台字体兼容

# --- 1. Data Acquisition (Real Data) ---
ticker_symbol = "NVDA"
print(f"Fetching institutional-grade data for {ticker_symbol}...")

# 下载数据
df = yf.download(ticker_symbol, period="5d", interval="1m", progress=False)

# [Best Practice] Data Cleaning Pipeline
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 确保索引是时间序列
df.index = pd.to_datetime(df.index)

# --- 2. Feature Engineering (Econophysics Metrics) ---
# 计算对数收益率 (Log Return) - 比简单收益率在数学上更严谨
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

# 代理波动率 (Local Volatility): 使用收益率的绝对值并标准化为基点 (Bps)
# 乘以 10000 是为了让数值易读 (e.g., 5 bps instead of 0.0005)
df['Local_Vol'] = df['Log_Ret'].abs() * 10000 

# [Best Practice] Binning Logic
# 使用 pd.cut 将价格切分为 60 个区间 (Bins)，模拟 Order Book 的 Price Levels
# 过滤掉极端的 Outliers (上下 1% 的价格)，防止图表被拉伸
price_lower = df['Close'].quantile(0.01)
price_upper = df['Close'].quantile(0.99)
df_clean = df[(df['Close'] >= price_lower) & (df['Close'] <= price_upper)].copy()

num_bins = 60
df_clean['Price_Bin'] = pd.cut(df_clean['Close'], bins=num_bins)

# --- 3. Aggregation (The Physics Engine) ---
# 统计每个价格区间的：总成交量 (阻力) 和 平均波动率 (速度)
profile = df_clean.groupby('Price_Bin', observed=True).agg({
    'Volume': 'sum',
    'Local_Vol': 'mean',
    'Close': 'mean' # 用区间均值作为 X 轴坐标
}).reset_index()

# 移除空桶
profile = profile[profile['Volume'] > 0]

# --- 4. Professional Visualization (Publication Layout) ---
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1]) # 上图高，下图矮

# === Plot A: The Phenomenon (Time-Price Dynamics) ===
ax1 = fig.add_subplot(gs[0])

# 绘制粒子密度 (成交量)
# [Pro Tip] 使用对数坐标 (log=True) 解决量级欺骗问题
bars = ax1.bar(range(len(profile)), profile['Volume'], color='#2E86C1', alpha=0.4, label='Particle Density (Liquidity)', width=0.8)
ax1.set_yscale('log') # 关键：开启对数坐标
ax1.set_ylabel("Liquidity Density (Volume, Log Scale)", color='#2E86C1', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#2E86C1')

# 格式化 Y 轴刻度 (例如 1M, 100k)
def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'B'][magnitude])
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))

# 绘制扩散速度 (波动率)
ax2 = ax1.twinx()
# 使用平滑曲线 (Cubic Spline 视觉效果更好，但这里用线性保持真实)
sns.lineplot(x=range(len(profile)), y=profile['Local_Vol'], ax=ax2, color='#E74C3C', linewidth=2.5, marker='o', markersize=4, label='Diffusion Speed (Volatility)')
ax2.set_ylabel("Volatility (Basis Points)", color='#E74C3C', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#E74C3C')

# X 轴美化
ax1.set_xticks(range(0, len(profile), 5)) # 每隔5个显示一个标签
ax1.set_xticklabels([f"{x:.1f}" for x in profile['Close'].iloc[::5]], rotation=0)
ax1.set_xlabel("Price Level ($)", fontweight='bold')
ax1.set_title(f"Market Microstructure Analysis: {ticker_symbol} (5-Day Intraday)\nEvidence of Liquidity Drag Effect", fontsize=16, fontweight='bold', pad=20)

# Legend 处理
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', frameon=True, fancybox=True, shadow=True)

# === Plot B: The Verification (Statistical Regression) ===
ax3 = fig.add_subplot(gs[1])

# [Pro Tip] 使用 Regression Plot 证明负相关性
# 同样使用 Log-Log 坐标，因为这符合幂律分布 (Power Law)
sns.regplot(x=np.log10(profile['Volume']), y=profile['Local_Vol'], ax=ax3, 
            scatter_kws={'alpha':0.5, 'color':'purple'}, line_kws={'color':'black'})

# 计算相关性系数
corr, p_value = stats.spearmanr(profile['Volume'], profile['Local_Vol'])

ax3.set_xlabel("Log10(Liquidity Density)", fontweight='bold')
ax3.set_ylabel("Volatility (Bps)", fontweight='bold')
ax3.set_title(f"Statistical Validation: Friction vs. Speed (Spearman Corr: {corr:.2f}, p-value: {p_value:.1e})", fontsize=12)

# 添加注释
ax3.text(0.02, 0.9, "Hypothesis: High Volume -> High Friction -> Low Volatility", transform=ax3.transAxes, fontsize=10, fontstyle='italic', bbox=dict(facecolor='white', alpha=0.8))

plt.show()

# --- 5. Output Interpretation for Social Media ---
print("="*50)
print("Quant Research Report Generated.")
print(f"1. Correlation Coefficient: {corr:.4f}")
print("2. Interpretation: " + ("Strong Friction Observed" if corr < -0.4 else "Weak Friction"))
print("="*50)