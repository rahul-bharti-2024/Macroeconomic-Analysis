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

# Output directories
resultsDir = "results"
modelsDir = "models"
os.makedirs(resultsDir, exist_ok=True)
os.makedirs(modelsDir, exist_ok=True)

# Load stationary dataset
stationaryDf = pd.read_csv("output_preprocess/df_stationary.csv", index_col=0)
stationaryDf.index = pd.to_datetime(stationaryDf.index)
stationaryDf = stationaryDf.sort_index().dropna()

if len(stationaryDf) < 40:
    raise RuntimeError("Insufficient data length for stable VAR training.")

# Train / val / test split
N = len(stationaryDf)
splitA = int(0.90 * N)
splitB = int(0.95 * N)

trainDf = stationaryDf.iloc[:splitA].copy()
valDf   = stationaryDf.iloc[splitA:splitB].copy()
testDf  = stationaryDf.iloc[splitB:].copy()

# Helper functions
def fitVarExplicit(trainFrame, lagOrder):
    model = VAR(trainFrame)
    res = model.fit(maxlags=lagOrder)
    if getattr(res, "k_ar", None) != lagOrder:
        res = model.fit(lagOrder)
    return res

def makeForecast(modelRes, trainFrame, targetIndex):
    initialWindow = trainFrame.values[-modelRes.k_ar:]
    pred = modelRes.forecast(initialWindow, steps=len(targetIndex))
    return pd.DataFrame(pred, index=targetIndex, columns=trainFrame.columns)

def getLastBlock(df, frac=0.4):
    total = len(df)
    blockSize = max(20, int(total * frac))
    return df.iloc[-blockSize:]

# Run models for every target variable
all_metrics = []

target_columns = list(stationaryDf.columns)

for target in target_columns:
    print(f"\n=== MODELING TARGET: {target} ===")

    #  FULL VAR 
    fullLag = 9
    t0 = time.perf_counter()
    fullModel = fitVarExplicit(trainDf, fullLag)
    time_full = time.perf_counter() - t0

    fullForecast = makeForecast(fullModel, trainDf, testDf.index)
    rmse_full = np.sqrt(mean_squared_error(testDf[target], fullForecast[target]))

    # save
    fullForecast[[target]].to_csv(os.path.join(resultsDir, f"{target}_forecast_full.csv"))
    with open(os.path.join(modelsDir, f"{target}_var_full.pkl"), "wb") as f:
        pickle.dump(fullModel, f)

    # CORESET VAR 
    coresetTrain = getLastBlock(trainDf, frac=0.4)
    coresetLag = 5

    t0 = time.perf_counter()
    coresetModel = fitVarExplicit(coresetTrain, coresetLag)
    time_coreset = time.perf_counter() - t0

    coresetForecast = makeForecast(coresetModel, coresetTrain, testDf.index)
    rmse_coreset = np.sqrt(mean_squared_error(testDf[target], coresetForecast[target]))

    # save
    coresetForecast[[target]].to_csv(os.path.join(resultsDir, f"{target}_forecast_coreset.csv"))
    with open(os.path.join(modelsDir, f"{target}_var_coreset.pkl"), "wb") as f:
        pickle.dump(coresetModel, f)

    # RANDOM PROJECTION VAR
    # Drop target so VAR uses only predictors
    predictors = trainDf.drop(columns=[target])
    if predictors.shape[1] >= 3:
        origDim = predictors.shape[1]
        projDim = min(8, max(3, origDim // 2))

        rp = GaussianRandomProjection(n_components=projDim, random_state=42)
        rpTrain = rp.fit_transform(predictors.values)
        rpCols = [f"RP_{i+1}" for i in range(projDim)]
        rpTrainDf = pd.DataFrame(rpTrain, index=predictors.index, columns=rpCols)

        rpLag = 5
        t0 = time.perf_counter()
        rpModel = fitVarExplicit(rpTrainDf, rpLag)
        time_scaled = time.perf_counter() - t0

        # project val/test
        rpTest = rp.transform(testDf.drop(columns=[target]).values)
        rpTestDf = pd.DataFrame(rpTest, index=testDf.index, columns=rpCols)

        # forecast RP space
        rpForecast = makeForecast(rpModel, rpTrainDf, testDf.index)
        
        # reconstruct target using linear regression
        reconModel = Ridge(alpha=1.0)
        reconModel.fit(rpTrainDf, trainDf[target])

        targetScaled = pd.Series(
            reconModel.predict(rpForecast),
            index=testDf.index, name=target
        )
        rmse_scaled = np.sqrt(mean_squared_error(testDf[target], targetScaled))

        # save
        pd.DataFrame({target: targetScaled}).to_csv(
            os.path.join(resultsDir, f"{target}_forecast_scaled.csv")
        )
        with open(os.path.join(modelsDir, f"{target}_var_scaled.pkl"), "wb") as f:
            pickle.dump(rpModel, f)
        with open(os.path.join(modelsDir, f"{target}_recon.pkl"), "wb") as f:
            pickle.dump(reconModel, f)

    else:
        # Not enough predictors for RP
        time_scaled = None
        rmse_scaled = None

    # Save comparison plot
    plt.figure(figsize=(11, 6))
    plt.plot(testDf.index, testDf[target], label="Actual", linewidth=2, color="black")
    plt.plot(fullForecast.index, fullForecast[target], label="Full VAR", linestyle="--")
    plt.plot(coresetForecast.index, coresetForecast[target], label="Coreset VAR", linestyle="-.")

    if rmse_scaled is not None:
        plt.plot(targetScaled.index, targetScaled, label="Scaled VAR (RP)", linestyle=":")

    plt.xlabel("Date")
    plt.ylabel(f"{target} (stationary)")
    plt.title(f"Forecast Comparison for {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(resultsDir, f"{target}_forecast_plot.png"), dpi=150)
    plt.close()

    # Add metrics row
    all_metrics.append({
        "target": target,
        "rmse_full": rmse_full,
        "rmse_coreset": rmse_coreset,
        "rmse_scaled": rmse_scaled,
        "time_full": time_full,
        "time_coreset": time_coreset,
        "time_scaled": time_scaled
    })

# Save all metrics
df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv(os.path.join(resultsDir, "metrics_all.csv"), index=False)

print("\nModeling complete for ALL TARGET VARIABLES.")
print(df_metrics)
