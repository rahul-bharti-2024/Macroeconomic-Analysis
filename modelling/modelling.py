import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.random_projection import GaussianRandomProjection

# Output directory setup
resultsDir = "results"
modelsDir = "models"
os.makedirs(resultsDir, exist_ok=True)
os.makedirs(modelsDir, exist_ok=True)

# Load the stationary data prepared earlier
stationaryDf = pd.read_csv("output_preprocess/df_stationary.csv", index_col=0)
stationaryDf.index = pd.to_datetime(stationaryDf.index)
stationaryDf = stationaryDf.sort_index().dropna()

# Ensure dataset is large enough
dataCount = len(stationaryDf)
if dataCount < 40:
    raise RuntimeError("Insufficient data length for stable VAR training.")

# Train/validation/test segmentation
splitA = int(0.9 * dataCount)
splitB = int(0.95 * dataCount)

trainDf = stationaryDf.iloc[:splitA].copy()
valDf   = stationaryDf.iloc[splitA:splitB].copy()
testDf  = stationaryDf.iloc[splitB:].copy()

def fitVarExplicit(trainFrame, lagOrder):
    """Fit VAR while enforcing a specific lag-order."""
    model = VAR(trainFrame)
    res = model.fit(maxlags=lagOrder)
    if getattr(res, "k_ar", None) != lagOrder:
        res = model.fit(lagOrder)
    return res

def makeForecast(modelRes, trainFrame, targetIndex):
    """Generate forecasts aligned to the target index period."""
    initialWindow = trainFrame.values[-modelRes.k_ar:]
    pred = modelRes.forecast(initialWindow, steps=len(targetIndex))
    return pd.DataFrame(pred, index=targetIndex, columns=trainFrame.columns)

def trainUnrateReconstruction(featureFrame, fullFrame):
    """Fit a regression model to approximate UNRATE from RP features."""
    target = fullFrame["UNRATE"].loc[featureFrame.index]
    model = Ridge(alpha=1.0)
    model.fit(featureFrame, target)
    return model

def applyUnrateReconstruction(model, featFrame):
    """Use the trained regression model to predict stationary UNRATE."""
    return pd.Series(model.predict(featFrame), index=featFrame.index, name="UNRATE")

def getLastBlock(df, frac=0.4):
    """Take the last contiguous segment of the dataset (deterministic)."""
    total = len(df)
    blockSize = max(20, int(total * frac))
    start = max(0, total - blockSize)
    return df.iloc[start : start + blockSize]


# ---------------- FULL VAR ----------------
fullLag = 9
start = time.perf_counter()
fullModel = fitVarExplicit(trainDf, fullLag)
timeFull = time.perf_counter() - start

fullForecast = makeForecast(fullModel, trainDf, testDf.index)
rmseFull = np.sqrt(mean_squared_error(testDf["UNRATE"], fullForecast["UNRATE"]))

fullForecast.to_csv(os.path.join(resultsDir, "forecast_full.csv"))
with open(os.path.join(modelsDir, "var_full.pkl"), "wb") as f:
    pickle.dump(fullModel, f)


# ---------------- CORESET VAR ----------------
coresetTrain = getLastBlock(trainDf, frac=0.4)
coresetLag = 5

start = time.perf_counter()
coresetModel = fitVarExplicit(coresetTrain, coresetLag)
timeCoreset = time.perf_counter() - start

coresetForecast = makeForecast(coresetModel, coresetTrain, testDf.index)
rmseCoreset = np.sqrt(mean_squared_error(testDf["UNRATE"], coresetForecast["UNRATE"]))

coresetForecast.to_csv(os.path.join(resultsDir, "forecast_sampled.csv"))
with open(os.path.join(modelsDir, "var_sampled.pkl"), "wb") as f:
    pickle.dump(coresetModel, f)


# ---------------- RANDOM PROJECTION VAR ----------------
rpSource = trainDf.drop(columns=["UNRATE"], errors="ignore")
origDim = rpSource.shape[1]
projDim = min(8, max(3, origDim // 2))

rpObj = GaussianRandomProjection(n_components=projDim, random_state=42)
rpTrain = rpObj.fit_transform(rpSource.values)
rpCols = [f"RP_{i+1}" for i in range(rpTrain.shape[1])]

rpTrainDf = pd.DataFrame(rpTrain, index=rpSource.index, columns=rpCols)

rpValDf = pd.DataFrame(rpObj.transform(valDf.drop(columns=["UNRATE"], errors="ignore")),
                       index=valDf.index, columns=rpCols)
rpTestDf = pd.DataFrame(rpObj.transform(testDf.drop(columns=["UNRATE"], errors="ignore")),
                        index=testDf.index, columns=rpCols)

reconModel = trainUnrateReconstruction(rpTrainDf, trainDf)

rpLag = 5

start = time.perf_counter()
scaledVarModel = fitVarExplicit(rpTrainDf, rpLag)
timeScaled = time.perf_counter() - start

rpForecast = makeForecast(scaledVarModel, rpTrainDf, testDf.index)
scaledUnrate = applyUnrateReconstruction(reconModel, rpForecast)
rmseScaled = np.sqrt(mean_squared_error(testDf["UNRATE"], scaledUnrate))

pd.DataFrame({"UNRATE": scaledUnrate}).to_csv(os.path.join(resultsDir, "forecast_scaled.csv"))

with open(os.path.join(modelsDir, "var_scaled.pkl"), "wb") as f:
    pickle.dump(scaledVarModel, f)
with open(os.path.join(modelsDir, "recon_rp.pkl"), "wb") as f:
    pickle.dump(reconModel, f)


metrics = {
    "rmseFull": rmseFull,
    "rmseCoreset": rmseCoreset,
    "rmseScaled": rmseScaled,
    "timeFull": timeFull,
    "timeCoreset": timeCoreset,
    "timeScaled": timeScaled,
}

pd.Series(metrics).to_csv(os.path.join(resultsDir, "metrics.csv"))


plt.figure(figsize=(11, 6))
plt.plot(testDf.index, testDf["UNRATE"], label="Actual (stationary)", linewidth=2, color="black")
plt.plot(fullForecast.index, fullForecast["UNRATE"], label="Full VAR", linestyle="--")
plt.plot(scaledUnrate.index, scaledUnrate, label="Scaled VAR (RP)", linestyle=":")
plt.plot(coresetForecast.index, coresetForecast["UNRATE"], label="Coreset VAR", linestyle="-.")
plt.xlabel("Date")
plt.ylabel("UNRATE (stationary)")
plt.legend()
plt.title("Forecast Comparison: Full vs Scaled vs Coreset VAR")
plt.tight_layout()
plt.savefig(os.path.join(resultsDir, "forecast_compare.png"), dpi=150)
plt.close()

print("Modeling complete. Metrics:")
print(pd.Series(metrics))
