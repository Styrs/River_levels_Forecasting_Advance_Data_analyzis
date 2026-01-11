import os
import glob
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

############################################################################################################
# function to merge result tables from different models into single tables per type
############################################################################################################



def merge_results_together(Config):
    # -----------------------------
    # User settings
    # -----------------------------
    Config = Config  # "A", "B", "C", "D"
    base_results_dir = "Results"
    config_dir = os.path.join(base_results_dir, f"Config_{Config}/individual_results")

    # The 4 csv "types" you want to merge
    TYPES = [
        "results",
        "per_horizon",
        "test_results",
        "test_per_horizon",
    ]

    # -----------------------------
    # Helpers
    # -----------------------------
    def infer_model_from_filename(filename: str) -> str:
        """Infer model name from a filename (trusted naming convention)."""
        name = os.path.basename(filename).lower()

        if "lstm" in name:
            return "LSTM"
        if "gru" in name:
            return "GRU"
        if "tree" in name:
            return "Tree"
        if "xgb" in name:
            return "XGBoost"

        return "UNKNOWN"

    def merge_one_type(csv_type: str) -> str:
        """
        Merge all model CSVs of a given type (stack rows), add 'model' column,
        and write output into merged_<type>/merged_<type>_<Config>.csv.

        Returns the output filepath.
        """
        # Example pattern: results_*B.csv
        pattern = os.path.join(config_dir, f"{csv_type}_*{Config}.csv")
        files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No files found for type='{csv_type}' with pattern: {pattern}")

        dfs = []
        for fp in files:
            df = pd.read_csv(fp, sep=",")
            df.insert(0, "model", infer_model_from_filename(fp))  # put model as first column
            dfs.append(df)

        merged = pd.concat(dfs, axis=0, ignore_index=True)

        out_dir = os.path.join(config_dir, f"merged_{csv_type}")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"merged_{csv_type}_{Config}.csv")
        merged.to_csv(out_path, index=False)

        return out_path

    # -----------------------------
    # execute the code
    # -----------------------------
    
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Config folder not found: {config_dir}")

    print(f"Config folder: {config_dir}")

    for t in TYPES:
        out = merge_one_type(t)
        print(f"[OK] merged '{t}' -> {out}")



############################################################################################################
# Test the OLS on horizon MSEs
############################################################################################################



def run_horizon_ols(
    df: pd.DataFrame,
    mse_col: str = "MSE",
    horizon_col: str = "horizon",
    model_col: str = "model",
    robust: bool = True
):
    """
    Runs OLS regressions of MSE on horizon for each model separately.

    Returns
    -------
    results_df : pd.DataFrame
        One row per model with:
        - beta (slope)
        - std_error
        - t_stat
        - p_value
        - r_squared
        - n_obs
    """

    results = []

    for model_name, g in df.groupby(model_col):

        y = g[mse_col].astype(float)
        X = sm.add_constant(g[horizon_col].astype(float))

        ols = sm.OLS(y, X)
        res = ols.fit(cov_type="HC1") if robust else ols.fit()

        results.append({
            "model": model_name,
            "beta_horizon": res.params[horizon_col],
            "std_error": res.bse[horizon_col],
            "t_stat": res.tvalues[horizon_col],
            "p_value": res.pvalues[horizon_col],
            "r_squared": res.rsquared,
            "n_obs": int(res.nobs),
        })

    return pd.DataFrame(results)



def analyze_config(
    path_to_csv: str,
    config_name: str
):
    """
    Loads merged_test_per_horizon_[A|B].csv and runs horizon OLS.

    We only have RMSE in the saved files, so we rebuild MSE as:
        MSE = RMSE^2
    """

    df = pd.read_csv(path_to_csv)

    # Sanity checks (your file has RMSE, not MSE)
    required_cols = {"model", "horizon", "RMSE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Compute MSE from RMSE for consistency, should have been done in the file directely but no time now
    df["MSE"] = df["RMSE"].astype(float) ** 2

    # Run OLS on MSE
    results_df = run_horizon_ols(
        df,
        mse_col="MSE",          # IMPORTANT: use the computed MSE
        horizon_col="horizon",
        model_col="model",
        robust=True
    )

    results_df["config"] = config_name
    return results_df



############################################################################################################
# Hyperparameter OLS: effect of hyperparameters on global_mse and var_mse_across_horizons
############################################################################################################

def run_hyperparam_ols_for_config(
    config_name: str,
    path_to_csv: str,
    out_root: str = "Results",
    robust: bool = True,
):
    """
    Runs 8 blocks total across configs when called twice (A,B):
      For each model in {Tree, XGBoost, LSTM, GRU}:
        - OLS(global_mse ~ const + hyperparams)
        - OLS(var_mse_across_horizons ~ const + hyperparams)

    Assumptions (your pipeline):
      - NAs are structural across models only.
      - Within a model, its relevant hyperparameter columns are filled.
    """

    df = pd.read_csv(path_to_csv)

    # Minimal required columns
    required = {"model", "global_mse", "var_mse_across_horizons"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path_to_csv}: {missing}")

    out_dir = os.path.join(out_root, "hyperparam_ols", f"Config_{config_name}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Explicit regressor sets per model (readable + no NA gymnastics) ---
    model_X_levels = {
        "Tree":    ["n_estimators", "max_depth", "min_samples_leaf"],
        "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
        "LSTM":    ["learning_rate", "batch_size", "epochs", "patience", "units", "num_layers", "dropout", "recurrent_dropout"],
        "GRU":     ["learning_rate", "batch_size", "epochs", "patience", "units", "num_layers", "dropout", "recurrent_dropout"],
    }

    # Which ones we log-transform (kept explicit and consistent)
    log_vars = {
        "learning_rate",
        "batch_size",
        "epochs",
        "patience",
        "units",
        "n_estimators",
        "min_samples_leaf",
    }

    def _add_logs_inplace(d: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
        """
        Create log_<col> for those in log_vars; return final X column names.
        Minimal validation: if non-positive -> raise.
        """
        X_cols_final = []
        logged_used = []

        for c in cols:
            if c in log_vars:
                x = pd.to_numeric(d[c], errors="coerce")
                if (x <= 0).any():
                    raise ValueError(f"Cannot log-transform '{c}' because it has non-positive values in config {config_name}.")
                d[f"log_{c}"] = np.log(x.astype(float))
                X_cols_final.append(f"log_{c}")
                logged_used.append(c)
            else:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                X_cols_final.append(c)

        return d, logged_used, X_cols_final

    def _fit_one(df_sub: pd.DataFrame, y_col: str, X_cols_levels: list[str], model_name: str):
        """
        Fit OLS with robust SE (HC1) and save summary.
        """
        d = df_sub.copy()

        # Build X with logs where needed
        d, logged_used, X_cols = _add_logs_inplace(d, X_cols_levels)

        # Keep only needed columns, drop missing (should not drop in your controlled setup)
        cols_needed = [y_col] + X_cols
        d = d[cols_needed].dropna(axis=0, how="any")

        y = d[y_col].astype(float)
        X = sm.add_constant(d[X_cols].astype(float), has_constant="add")

        res = sm.OLS(y, X).fit(cov_type="HC1") if robust else sm.OLS(y, X).fit()

        # Save text summary
        summary_path = os.path.join(out_dir, f"summary_{y_col}_{model_name}_{config_name}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Config: {config_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dependent variable: {y_col}\n")
            f.write(f"Regressors (final): {list(X.columns)}\n")
            f.write(f"Logged variables used (original): {logged_used}\n")
            f.write(f"Robust SE: {'HC1' if robust else 'None'}\n\n")
            f.write(res.summary().as_text())

        # Long-form rows
        rows = []
        for term in res.params.index:
            rows.append({
                "config": config_name,
                "model": model_name,
                "y": y_col,
                "term": term,
                "coef": float(res.params[term]),
                "std_error": float(res.bse[term]),
                "t_stat": float(res.tvalues[term]),
                "p_value": float(res.pvalues[term]),
                "r_squared": float(res.rsquared),
                "n_obs": int(res.nobs),
                "logged_vars_used": ",".join(logged_used),
            })
        return rows, res

    all_rows = []

    for model_name, X_cols_levels in model_X_levels.items():
        df_m = df[df["model"] == model_name].copy()

        if df_m.empty:
            print(f"[WARN] No rows for model={model_name} in config {config_name}")
            continue

        for y_col in ["global_mse", "var_mse_across_horizons"]:
            rows, res = _fit_one(df_m, y_col, X_cols_levels, model_name)
            all_rows.extend(rows)
            print(f"[OK] Hyperparam OLS done: config={config_name}, model={model_name}, y={y_col}, n={int(res.nobs)}, k={len(res.params)}")

    out_csv = os.path.join(out_dir, f"hyperparam_ols_long_{config_name}.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved long-form coefficient table -> {out_csv}")

    # Save the explicit list of log variables (as requested)
    note_path = os.path.join(out_dir, f"log_transforms_{config_name}.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("Log-transformed variables (when used by a model):\n")
        for v in sorted(log_vars):
            f.write(f"- {v}\n")
    print(f"[OK] Saved log transform note -> {note_path}")

    return out_csv



################################################################################
# Plots for the best models
################################################################################


def plot_best_models_MSE_per_horizon(Config):
    """
    Plot MSE per horizon for the best models in Config {Config}.
    Assumes the file 'test_per_horizon_best_models_ConfigA.csv' exists.
    """

    df = pd.read_csv(f"Results/Config_{Config}/merged_test_per_horizon/merged_test_per_horizon_{Config}.csv")
    df["MSE"] = df["RMSE"] ** 2

    plt.figure()

    for model_name, g in df.groupby("model"):
        plt.plot(g["horizon"], g["MSE"], label=model_name)

    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title(f"MSE per Horizon — Best Models (Config {Config})")
    plt.legend()
    plt.show()

def plot_best_models_R2_per_horizon(Config):
    """
    Plot R² per horizon for the best models in Config {Config}.
    Assumes the file 'merged_test_per_horizon_{Config}.csv' exists.
    """

    df = pd.read_csv(
        f"Results/Config_{Config}/merged_test_per_horizon/merged_test_per_horizon_{Config}.csv"
    )

    plt.figure()

    for model_name, g in df.groupby("model"):
        plt.plot(g["horizon"], g["R2"], label=model_name)

    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("R²")
    plt.title(f"R² per Horizon — Best Models (Config {Config})")
    plt.legend()
    plt.show()


def plot_best_models_historgram (variable = "global_mse"):
    df_A = pd.read_csv("Results/Config_A/merged_test_results/merged_test_results_A.csv")
    df_B = pd.read_csv("Results/Config_B/merged_test_results/merged_test_results_B.csv")

    df_A["config"] = "A"
    df_B["config"] = "B"

    df = pd.concat([df_A, df_B], ignore_index=True)

    # Ensure consistent model order
    model_order = ["LSTM", "GRU", "Tree", "XGBoost"]

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model")

    # --------------------------------------------------
    # Prepare bar positions
    # --------------------------------------------------
    x = np.arange(len(model_order))
    width = 0.35

    res_A = df[df["config"] == "A"][f"test_{variable}"].values
    res_B = df[df["config"] == "B"][f"test_{variable}"].values
    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure()

    plt.bar(x - width/2, res_A, width, label="Config A")
    plt.bar(x + width/2, res_B, width, label="Config B")

    plt.xticks(x, model_order)
    plt.ylabel(f"Global Test {variable.replace('_', ' ').title()}")
    plt.xlabel("Model")
    plt.title(f"Global Test {variable.replace('_', ' ').title()} — Best Models (Config A vs B)")
    plt.legend()
    plt.tight_layout()
    plt.show()
