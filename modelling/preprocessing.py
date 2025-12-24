import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller
from sklearn.random_projection import GaussianRandomProjection
import pyfredapi as pf

load_dotenv()
os.environ["FRED_API_KEY"] = str(os.getenv("API_KEY", ""))

OUTPUT_DIR = "output_preprocess"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. DOWNLOAD RAW DATA FROM FRED
series_ids = {
    "UNRATE": "UNRATE",
    "PCE": "PCE",
    "CPI": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "M2": "M2SL",
    "WAGE_AVG": "CES0500000003",
    "INDPRO": "INDPRO",
    "GDPDEF": "GDPDEF",
    "CSENT": "UMCSENT",
    "CRUDE": "DCOILWTICO",
    "COMPI": "PPIACO",
    "LABOR": "CIVPART"
}

dfs = []
for name, sid in series_ids.items():
    df = pf.get_series(series_id=sid)
    df = df.rename(columns={"value": name})
    df.drop(columns=["realtime_start", "realtime_end"], inplace=True)
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    dfs.append(df)

# 2. ALIGN TO COMMON MONTHLY INDEX
start = max(df.index.min() for df in dfs)
end = min(df.index.max() for df in dfs)
all_months = pd.date_range(start=start, end=end, freq="ME")

aligned = []
for df in dfs:
    delta = df.index[1] - df.index[0]
    if delta <= pd.Timedelta(days=31):
        df = df.resample("ME").mean()
    else:
        df = df.resample("ME").ffill()
    aligned.append(df.reindex(all_months))

# Combine
df_all = pd.concat(aligned, axis=1)
df_all = df_all.astype(float)
df_all.to_csv(os.path.join(OUTPUT_DIR, "df_all.csv"))

# 3. STATIONARIZE DATASET
# -----------------------------
list_highval = ["PCE", "M2", "WAGE_AVG", "INDPRO", "CRUDE", "COMPI"]
df_stationary = pd.DataFrame(index=df_all.index)

for col in df_all.columns:
    s = df_all[col].copy()
    if col in list_highval:
        s = s.replace({0: np.nan}).dropna()
        s = np.log(s)

    p = adfuller(s.dropna())[1]
    diffs = 0
    while p > 0.05 and diffs < 2:
        s = s.diff().dropna()
        diffs += 1
        try:
            p = adfuller(s.dropna())[1]
        except:
            break

    df_stationary[col] = s

# drop rows with NaN due to differencing
df_stationary = df_stationary.dropna()
df_stationary.to_csv(os.path.join(OUTPUT_DIR, "df_stationary.csv"))

# -----------------------------
# 4. RANDOM PROJECTION (SCALING TECHNIQUE)
# -----------------------------
orig_d = df_stationary.shape[1]
proj_k = min(8, max(3, orig_d // 2))

rp = GaussianRandomProjection(n_components=proj_k, random_state=42)
X_rp = rp.fit_transform(df_stationary.values)

rp_cols = [f"RP_{i+1}" for i in range(proj_k)]
df_scaled = pd.DataFrame(X_rp, index=df_stationary.index, columns=rp_cols)
df_scaled.to_csv(os.path.join(OUTPUT_DIR, "df_scaled.csv"))

print("Preprocessing complete.")
