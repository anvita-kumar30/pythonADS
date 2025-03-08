import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, f_oneway, zscore

# Load dataset
df = pd.read_csv('diamonds.csv')

# ---- Population Statistics ----
population_mean = df['price'].mean()
population_std_dev = df['price'].std()

# ---- Hypothesis Testing ----

# One-Sample T-Test: Compare mean price to hypothesis value (H0: μ = 3200)
hypothesis_mean = 3200
t_stat_1samp, p_val_1samp = ttest_1samp(df['price'], hypothesis_mean)

# Independent T-Test: Compare mean price of diamonds above and below 1 carat
above_1carat = df[df['carat'] > 1]['price']
below_1carat = df[df['carat'] <= 1]['price']
t_stat, p_val = ttest_ind(above_1carat, below_1carat, equal_var=False)

# ANOVA Test: Compare mean prices across different cut categories (all categories)
anova_stat, anova_p_val = f_oneway(
    *[df[df['cut'] == cut]['price'] for cut in df['cut'].unique()]
)

# ---- Sample-Based Z-Test and T-Test ----
sample_size = 40
sample = df.sample(n=sample_size, random_state=42)

sample_mean = sample['price'].mean()
sample_std_dev = sample['price'].std()

# Z-Test Formula: (X̄ - μ) / (σ / sqrt(n))
Z_stat = (sample_mean - population_mean) / (population_std_dev / np.sqrt(sample_size))
Z_critical_value = 1.96  # 5% significance level (two-tailed)

# T-Test for small sample (n < 30)
sample_small = df.sample(n=20, random_state=42)
sample_small_mean = sample_small['price'].mean()
sample_small_std_dev = sample_small['price'].std()
t_stat_small = (sample_small_mean - population_mean) / (sample_small_std_dev / np.sqrt(20))

# ---- Print Results ----
print("\nInferential Statistics:")

# One-Sample T-Test
print(f"One-Sample T-Test: t-stat: {t_stat_1samp:.3f}, p-value: {p_val_1samp:.3f}")
if p_val_1samp < 0.05:
    print("Reject H0: The mean price is significantly different from 3200.")
else:
    print("Fail to reject H0: No significant difference from 3200.")

# Independent T-Test
print(f"\nT-Test Between Carat Groups: t-stat: {t_stat:.3f}, p-value: {p_val:.3f}")
if p_val < 0.05:
    print("Reject H0: The price difference between carat groups is significant.")
else:
    print("Fail to reject H0: No significant price difference.")

# ANOVA Test
print(f"\nANOVA Test: F-stat: {anova_stat:.3f}, p-value: {anova_p_val:.3f}")
if anova_p_val < 0.05:
    print("Reject H0: There is a significant price difference across cut types.")
else:
    print("Fail to reject H0: No significant price difference across cut types.")

# Z-Test Interpretation
print(f"\nZ-Test: Z-stat: {Z_stat:.3f}, Critical Z: ±{Z_critical_value}")
if abs(Z_stat) > Z_critical_value:
    print("Reject H0: The sample price mean is significantly different from population mean.")
else:
    print("Fail to reject H0: No significant difference in sample mean.")

# Small Sample T-Test
print(f"\nSmall Sample T-Test (n=20): t-stat: {t_stat_small:.3f}")
