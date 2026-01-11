import numpy as np
import pandas as pd
import data_clearing_function
import time
import tensorflow as tf
import math
import inspect, uuid, os



from typing import List, Dict, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import Optimizer

#############################################################################################################
# File preperation for the Ml models
#############################################################################################################


def prepare_station_dataframe(path_to_csv, Config):
    # 1 get the data -------------------------------------------------------------------------------------

    
    data_station = load_prepared_station_file(path_to_csv)


    # 2 Class column types -------------------------------------------------------------------------------

    all_cols = data_station.columns.tolist()
    slow_cols = [c for c in all_cols if c.endswith("_14d") or c.endswith("_30d")]
    nonslow_cols = [c for c in all_cols if c not in slow_cols]

        # Categorize non-slow columns

    self_non_slow = ["self_data"] if "self_data" in nonslow_cols else []

    seasonal_non_slow = [c for c in ["sin_week", "cos_week"] if c in nonslow_cols]

    time_non_slow = [c for c in ["Unix Date", "ISO Date"] if c in nonslow_cols]

    weather_non_slow = [c for c in nonslow_cols if c.startswith("W_")]

    water_non_slow = [c for c in nonslow_cols if c not in self_non_slow and c not in seasonal_non_slow and c not in time_non_slow and c not in weather_non_slow]

        # Categorize slow columns

    slow_self = [ c for c in slow_cols if c.startswith("self_data")]

    slow_weather = [c for c in slow_cols if c.startswith("W_")]

    slow_water = [ c for c in slow_cols if c not in slow_self and c not in slow_weather]

    # 3 Select columns based on configuration ----------------------------------------------------------

    if Config == "A":
        # Keep:
        # - self (non-slow)
        # - weather (non-slow)
        # - seasonal
        # - slow self
        # - slow weather
        # Remove:
        # - other rivers (non-slow)
        # - slow other rivers
        # - time columns
        
        keep_cols = ( self_non_slow +  weather_non_slow + seasonal_non_slow + slow_self + slow_weather)

    elif Config == "B":
        # Keep everything except time columns
        keep_cols = [c for c in all_cols if c not in time_non_slow]

    elif Config == "C":
        # Temporary: same as B (we will refine later)
        keep_cols = [c for c in all_cols if c not in time_non_slow]

    elif Config == "D":
        # Keep:
        # - self (non-slow)
        # - weather (non-slow)
        # - seasonal
        # - slow self
        # - slow weather
        # Optionally include other rivers depending on future decisions
        # For now: mirror A
        
        keep_cols = (self_non_slow +weather_non_slow +seasonal_non_slow +slow_self +slow_weather)

    else:
        raise ValueError("CONFIG must be one of: 'A', 'B', 'C', 'D'.")

    # 4 Remove unwanted columns -------------------------------------------------------------------------

    column_list_to_remove = [ c for c in all_cols if c not in keep_cols]

    prepared_config_data_station = data_clearing_function.remove_columns(data_station,column_list_to_remove,True)
    print("Column remove and ready to go")
    
    return prepared_config_data_station


def load_prepared_station_file(path_to_csv) :
    """
    Load a prepared CSV for a single water station and do basic preprocessing.

    Steps:
    - Read the CSV.
    - Sort rows by 'Unix Date' (time axis).
    

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the prepared station CSV file.
    

    Returns
    -------
    df : pandas.DataFrame
        Loaded and time-sorted dataframe.
    """

    # Load raw data
    df = pd.read_csv(path_to_csv,sep=",", na_values=["NAN", "NaN", "nan", "NA", ""])

    # Sort by time (we assume hourly consistency is already guaranteed)
    df = df.sort_values("Unix Date").reset_index(drop=True)

    return df


def make_windowed_data(df, target_col="self_data", lookback=72, horizon=72):
    """
    Transform a time series DataFrame into a supervised learning dataset.

    Each sample uses the previous `lookback` rows of ALL columns as features,
    and the next `horizon` values of `target_col` as targets.

    Parameters
    ----------
    df : pandas.DataFrame
        Time-sorted DataFrame containing the target column and all feature columns.
    target_col : str, optional (default 'self_data')
        Name of the target column to forecast.
    lookback : int
        Number of past time steps to use as input features.
    horizon : int
        Number of future time steps to predict.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, lookback * n_features).
    y : numpy.ndarray
        Target matrix of shape (n_samples, horizon).
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    values = df.values  # shape: (n_obs, n_features)
    target = df[target_col].values  # shape: (n_obs,)

    n_obs = len(df)
    n_features = df.shape[1]

    X_list = []
    y_list = []

    # Last possible index for the start of the prediction window
    max_start = n_obs - horizon

    for end_input in range(lookback, max_start):
        start_input = end_input - lookback
        # Input: previous `lookback` rows of all features, flattened
        X_window = values[start_input:end_input, :]          # shape: (lookback, n_features)
        X_list.append(X_window.reshape(-1))                  # flatten to (lookback * n_features,)

        # Output: next `horizon` steps of the target
        y_window = target[end_input:end_input + horizon]     # shape: (horizon,)
        y_list.append(y_window)

    if not X_list:
        raise ValueError(
            "Not enough data to create supervised samples with "
            f"lookback={lookback} and horizon={horizon}."
        )

    X = np.vstack(X_list)   # (n_samples, lookback * n_features)
    y = np.vstack(y_list)   # (n_samples, horizon)

    return X, y


def make_windowed_data_NN(df: pd.DataFrame,target_col: str = "self_data",lookback: int = 72,horizon: int = 72):
    """
    Transform a time series DataFrame into a supervised learning dataset
    formatted for neural networks / LSTMs.

    Each sample uses the previous `lookback` rows of ALL columns as inputs
    (kept as a 2D time × features block), and the next `horizon` values
    of `target_col` as outputs.

    Parameters
    ----------
    df : pandas.DataFrame
        Time-sorted DataFrame containing the target column and all feature
        columns you want to use (after applying your A/B/C/D config).
    target_col : str, optional (default 'self_data')
        Name of the target column to forecast.
    lookback : int
        Number of past time steps to use as input features.
    horizon : int
        Number of future time steps to predict.

    Returns
    -------
    X : numpy.ndarray
        Input tensor of shape (n_samples, lookback, n_features),
        suitable for LSTM / RNN models.
    y : numpy.ndarray
        Target matrix of shape (n_samples, horizon).
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Convert to numpy arrays
    values = df.values          # shape: (n_obs, n_features)
    target = df[target_col].values  # shape: (n_obs,)

    n_obs = len(df)
    n_features = df.shape[1]

    X_list = []
    y_list = []

    # Last possible index for the start of the prediction window
    max_start = n_obs - horizon

    for end_input in range(lookback, max_start):
        start_input = end_input - lookback

        # Input: previous `lookback` rows of all features, as a 2D block
        X_window = values[start_input:end_input, :]         # (lookback, n_features)
        X_list.append(X_window)

        # Output: next `horizon` steps of the target
        y_window = target[end_input:end_input + horizon]    # (horizon,)
        y_list.append(y_window)

    if not X_list:
        raise ValueError(
            "Not enough data to create supervised samples with "
            f"lookback={lookback} and horizon={horizon}."
        )

    # Stack into final arrays
    X = np.stack(X_list, axis=0)   # (n_samples, lookback, n_features)
    y = np.vstack(y_list)          # (n_samples, horizon)

    return X, y


def split_time_series(X, y, train_frac=0.7, val_frac=0.2):
    """
    Split time series data into train, validation, and test sets
    without shuffling (respect time order).

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target matrix of shape (n_samples, horizon) or (n_samples,).
    train_frac : float, default 0.7
        Fraction of samples to use for training.
    val_frac : float, default 0.2
        Fraction of samples to use for validation.

    Returns
    -------
    X_train, y_train
    X_val, y_val
    X_test, y_test
    """

    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples.")

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    n_test = n_samples - n_train - n_val

    if n_test <= 0:
        raise ValueError("Fractions leave no samples for test set.")

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


#############################################################################################################
# ML model training and evaluation functions
#############################################################################################################

#----LSTM model functions -----

def build_lstm_model(
    input_shape: Tuple[int, int],
    output_dim: int,
    params: Dict
) -> tf.keras.Model:
    """
    Build a multi-output LSTM model (sequence -> vector).

    Parameters
    ----------
    input_shape : tuple
        (timesteps, n_features)
    output_dim : int
        Number of outputs (e.g. forecast horizon, 72)
    params : dict
        Hyperparameters for the model. Expected keys (with defaults):
        - units: int
        - num_layers: int (1 or 2 typical)
        - dropout: float
        - recurrent_dropout: float
        - learning_rate: float

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model.
    """
    units = params.get("units", 64)
    num_layers = params.get("num_layers", 1)
    dropout = params.get("dropout", 0.0)
    recurrent_dropout = params.get("recurrent_dropout", 0.0)
    learning_rate = params.get("learning_rate", 1e-3)

    model = Sequential()
    
    # First / only LSTM layer
    if num_layers == 1:
        model.add(
            LSTM(
                units,
                input_shape=input_shape,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
    else:
        # First LSTM with return_sequences=True
        model.add(
            LSTM(
                units,
                input_shape=input_shape,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
        # Second / last LSTM
        model.add(
            LSTM(
                units,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )

    # Output layer: 72-hour forecast (or whatever y.shape[1] is)
    model.add(Dense(output_dim))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

    return model


def find_best_LSTM_model(
    X_train,
    y_train,
    X_val,
    y_val,
    param_scope,
    n_random_start=5,
    n_echo=20,
    random_state=0,
):
    """
    Random + Bayesian hyperparameter search for the LSTM model.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs, shape (n_samples, timesteps, n_features).
    y_train : np.ndarray
        Training targets, shape (n_samples, horizon).
    X_val : np.ndarray
        Validation inputs, shape (n_val_samples, timesteps, n_features).
    y_val : np.ndarray
        Validation targets, shape (n_val_samples, horizon).
    param_scope : dict
        Hyperparameter search space for LSTM, same structure as for RF/XGB:
        {
            "units": {"type": "int", "bounds": (8, 128)},
            ...
        }
    n_random_start : int
        Number of pure random search iterations before Bayesian phase.
    n_echo : int
        Number of Bayesian iterations (new configs) to evaluate.
    random_state : int
        Seed for numpy's random generator (for reproducibility).

    Returns
    -------
    best_model : tf.keras.Model or None
        Best LSTM model found (by global MSE on validation).
    best_params : dict or None
        Best hyperparameter configuration.
    best_global_mse : float
        Lowest global MSE reached on validation.
    all_results : list of dict
        Detailed info for each evaluated configuration, including:
        - "iteration_id"
        - "strategy": "random" or "bayes"
        - "params": hyperparameter dict
        - "model": trained model
        - "global_mse"
        - "global_r2"
        - "per_horizon_metrics": DataFrame with metrics per horizon
        - "varianceofMSE_across_horizons"
    results_df : pd.DataFrame
        2D summary DataFrame with one row per run (global metrics).
    per_horizon_df : pd.DataFrame
        Per-horizon metrics for each run (2D, with model_id / iteration_id).
    """

    # ------------------------------------------------------------------
    # Logging / call context (same style as RF / XGB / GRU)
    # ------------------------------------------------------------------
    call_id = uuid.uuid4().hex[:8]

    stack = inspect.stack()
    caller = stack[1]  # caller frame
    caller_file = caller.filename
    caller_line = caller.lineno
    caller_function = caller.function

    print(f"\n==== NEW CALL TO find_best_LSTM_model ====")
    print(f"call_id={call_id}, pid={os.getpid()}")
    print(f"Called from: file='{caller_file}', line={caller_line}, function='{caller_function}'\n")

    # ------------------------------------------------------------------
    # State containers
    # ------------------------------------------------------------------
    all_results = []
    best_model = None
    best_params = None
    best_global_mse = +np.inf

    # For building models
    input_shape = X_train.shape[1:]   # (timesteps, n_features)
    output_dim = y_train.shape[1]     # horizon

    # ------------------------------------------------------------------
    # Helper to sample random hyperparameters from param_scope
    # ------------------------------------------------------------------
    def sample_params(param_scope: Dict[str, Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Sample a random hyperparameter configuration from the given scope.
        Supports types: "int", "float", "categorical".
        If "prior" == "log-uniform" for floats, sample in log-space.
        """
        params = {}
        for name, spec in param_scope.items():
            ptype = spec.get("type", "float")
            if ptype == "int":
                low, high = spec["bounds"]
                # rng.integers is [low, high), so we add +1 to include high
                val = int(rng.integers(low, high + 1))
            elif ptype == "float":
                low, high = spec["bounds"]
                prior = spec.get("prior", "uniform")
                if prior == "log-uniform":
                    log_low, log_high = np.log(low), np.log(high)
                    val = float(np.exp(rng.uniform(log_low, log_high)))
                else:
                    val = float(rng.uniform(low, high))
            elif ptype == "categorical":
                values = spec["values"]
                idx = rng.integers(0, len(values))
                val = values[idx]
            else:
                raise ValueError(f"Unsupported param type '{ptype}' for '{name}'")

            params[name] = val
        return params

    # ------------------------------------------------------------------
    # Random search phase
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=random_state)

    print("=== Starting LSTM random search phase ===")
    for i in range(n_random_start):
        print(f"\n--- Random Search iteration {i+1}/{n_random_start} ---")

        params = sample_params(param_scope, rng)
        print("Sampled params:", params)

        # Ensure types are consistent
        units = int(params.get("units", 64))
        num_layers = int(params.get("num_layers", 1))
        dropout = float(params.get("dropout", 0.1))
        recurrent_dropout = float(params.get("recurrent_dropout", 0.1))
        batch_size = int(params.get("batch_size", 64))
        epochs = int(params.get("epochs", 50))
        patience = int(params.get("patience", 5))
        learning_rate = float(params.get("learning_rate", 1e-3))

        # Rebuild a params dict that we pass into build_lstm_model
        model_params = {
            "units": units,
            "num_layers": num_layers,
            "dropout": dropout,
            "recurrent_dropout": recurrent_dropout,
            "learning_rate": learning_rate,
        }

        # Build LSTM model
        model = build_lstm_model(input_shape, output_dim, model_params)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )

        # Train
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Predict and evaluate
        y_val_pred = model.predict(X_val, verbose=0)

        global_mse = mean_squared_error(y_val, y_val_pred)
        global_r2 = r2_score(y_val.reshape(-1), y_val_pred.reshape(-1))

        per_horizon_df = evaluate_per_horizon(y_val, y_val_pred)
        var_mse = variance_per_horizon(per_horizon_df)

        print(f"Global validation MSE: {global_mse:.6f}")
        print(f"Global validation R^2: {global_r2:.6f}")
        print(f"Variance of MSE across horizons: {var_mse}")

        trial_result = {
            "iteration_id": i + 1,
            "strategy": "random",
            "params": params,   # keep original param dict including batch_size, epochs, patience
            "model": model,
            "global_mse": float(global_mse),
            "global_r2": global_r2,
            "per_horizon_metrics": per_horizon_df,
            "varianceofMSE_across_horizons": var_mse,
        }
        all_results.append(trial_result)

        if global_mse < best_global_mse:
            best_global_mse = global_mse
            best_model = model
            best_params = params

    # ------------------------------------------------------------------
    # Bayesian optimization phase (to be implemented separately)
    # ------------------------------------------------------------------
    print("\n>>> Starting Bayesian Optimization Phase (LSTM) <<<")
    best_model, best_params, best_global_mse, all_results = LSTM_baysian_hyperparam_optimization(
        best_model,
        best_params,
        best_global_mse,
        all_results,
        X_train,
        y_train,
        X_val,
        y_val,
        param_scope,
        n_echo,
        n_random_start,
    )

    # ------------------------------------------------------------------
    # Summarize all results
    # ------------------------------------------------------------------
    results_df, per_horizon_df = summarize_all_results_to_df(
        all_results,
        metric_key="global_mse",
    )

    return best_model, best_params, best_global_mse, all_results, results_df, per_horizon_df


def LSTM_baysian_hyperparam_optimization(
    best_model,
    best_params,
    best_global_mse,
    all_results,
    X_train,
    y_train,
    X_val,
    y_val,
    param_scope,
    n_echo,
    n_random_start,
):
    """
    Run Bayesian optimization for the LSTM model,
    starting from past runs stored in all_results.
    """
    print(f"[DEBUG LSTM BO] New call to LSTM_baysian_hyperparam_optimization")
    

    metric_key = "global_mse"
    random_state = 0  # keep consistent with other models

    if all_results is None:
        all_results = []

    # ==================================================================
    # 1) Small helpers used ONLY inside this function
    # ==================================================================

    def params_dict_to_vector(params, param_names):
        """Convert a param dict to a list/vector in fixed param_names order."""
        return [params[name] for name in param_names]

    def extract_Xy_from_results(all_results_local, param_names, metric_key_local="global_mse"):
        """
        Build X_seen and y_seen from past results for BO initialization.
        X_seen shape: (n_runs, n_params)
        y_seen shape: (n_runs,)
        """
        X_list = []
        y_list = []

        for res in all_results_local:
            if "params" not in res:
                continue
            if metric_key_local not in res:
                continue

            p = res["params"]
            # Keep only runs that have all required params
            if all(name in p for name in param_names):
                x_vec = params_dict_to_vector(p, param_names)
                y_val = res[metric_key_local]
                X_list.append(x_vec)
                y_list.append(y_val)

        if len(X_list) == 0:
            # No past observations
            return np.empty((0, len(param_names))), np.empty((0,))

        return np.array(X_list, dtype=float), np.array(y_list, dtype=float)

    def create_bayes_optimizer_from_results(
        all_results_local,
        param_scope_local,
        metric_key_local="global_mse",
        random_state_local=0,
        base_estimator="GP",
        acq_func="EI",
    ):
        """
        Prepare a skopt.Optimizer and feed it with past observations.
        This is a local copy, only used by LSTM_baysian_hyperparam_optimization.
        """
        # 1. Build skopt dimensions
        param_names_local = list(param_scope_local.keys())
        dimensions = []
        for name in param_names_local:
            spec = param_scope_local[name]
            ptype = spec.get("type", "float")

            if ptype == "int":
                low, high = spec["bounds"]
                dimensions.append(Integer(low, high, name=name))

            elif ptype == "float":
                low, high = spec["bounds"]
                prior = spec.get("prior", "uniform")
                if prior == "log-uniform":
                    dimensions.append(Real(low, high, prior="log-uniform", name=name))
                else:
                    dimensions.append(Real(low, high, prior="uniform", name=name))

            elif ptype == "categorical":
                values = spec["values"]
                dimensions.append(Categorical(values, name=name))

            else:
                raise ValueError(f"Unsupported parameter type '{ptype}' for '{name}'")

        # 2. Extract past observations
        X_seen_local, y_seen_local = extract_Xy_from_results(
            all_results_local,
            param_names_local,
            metric_key_local,
        )

        # 3. Create the optimizer
        optimizer_local = Optimizer(
            dimensions=dimensions,
            base_estimator=base_estimator,
            acq_func=acq_func,
            random_state=random_state_local,
            n_initial_points=0,  # we already have initial points from random search
        )

        # 4. Feed past observations (if any)
        if X_seen_local.shape[0] > 0:
            for x_vec, y_val in zip(X_seen_local, y_seen_local):
                optimizer_local.tell(list(x_vec), float(y_val))

        return optimizer_local, param_names_local, X_seen_local, y_seen_local

    # ==================================================================
    # 2) Train & evaluate ONE LSTM config (used inside BO loop)
    # ==================================================================
    def _train_and_evaluate_lstm_for_bayes(
        X_train_local,
        y_train_local,
        X_val_local,
        y_val_local,
        params,
    ):
        """Train an LSTM model with given params and compute validation metrics."""
        # Coerce values and extract training hyperparameters
        units = int(params.get("units", 64))
        num_layers = int(params.get("num_layers", 1))
        dropout = float(params.get("dropout", 0.1))
        recurrent_dropout = float(params.get("recurrent_dropout", 0.1))
        batch_size = int(params.get("batch_size", 64))
        epochs = int(params.get("epochs", 50))
        patience = int(params.get("patience", 5))
        learning_rate = float(params.get("learning_rate", 1e-3))

        # Input/output shapes (3D -> 2D)
        input_shape = X_train_local.shape[1:]   # (timesteps, n_features)
        output_dim = y_train_local.shape[1]     # horizon size

        # Params for the model-building function
        model_params = {
            "units": units,
            "num_layers": num_layers,
            "dropout": dropout,
            "recurrent_dropout": recurrent_dropout,
            "learning_rate": learning_rate,
        }

        # Build and train LSTM
        model = build_lstm_model(input_shape, output_dim, model_params)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            X_train_local,
            y_train_local,
            validation_data=(X_val_local, y_val_local),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Evaluate
        y_val_pred = model.predict(X_val_local, verbose=0)

        global_mse_local = mean_squared_error(y_val_local, y_val_pred)
        global_r2_local = r2_score(y_val_local.reshape(-1), y_val_pred.reshape(-1))
        per_horizon_df_local = evaluate_per_horizon(y_val_local, y_val_pred)

        return model, float(global_mse_local), float(global_r2_local), per_horizon_df_local

    # ==================================================================
    # 3) Main Bayesian loop (same structure as GRU)
    # ==================================================================

    optimizer, param_names, X_seen, y_seen = create_bayes_optimizer_from_results(
        all_results_local=all_results,
        param_scope_local=param_scope,
        metric_key_local=metric_key,
        random_state_local=random_state,
    )

    already_evaluated = X_seen.shape[0]
    total_planned = already_evaluated + n_echo

    print("\n=== Starting Bayesian optimization loop (LSTM) ===")
    print(f"Already evaluated models (random + previous BO): {already_evaluated}")
    print(f"Will run {n_echo} new Bayesian iterations.")
    print(f"Iteration indexing will start at {already_evaluated + 1} and go up to {total_planned}.")

    total_time = 0.0
    current_iter_index = already_evaluated

    for step in range(n_echo):
        iteration_id = current_iter_index + step + 1
        print(f"\n--- Bayesian iteration {iteration_id}/{total_planned} ---")

        # 1) Ask optimizer for a new candidate
        x_next = optimizer.ask()
        if not isinstance(x_next, (list, tuple, np.ndarray)):
            x_next = [x_next]

        # Map vector -> param dict
        params_next = {name: value for name, value in zip(param_names, x_next)}
        print("Proposed hyperparameters:", params_next)

        # 2) Train & evaluate LSTM with these params
        iter_start_time = time.time()
        model, global_mse, global_r2, per_horizon_df = _train_and_evaluate_lstm_for_bayes(
            X_train_local=X_train,
            y_train_local=y_train,
            X_val_local=X_val,
            y_val_local=y_val,
            params=params_next,
        )
        iter_duration = time.time() - iter_start_time
        total_time += iter_duration

        print(f"Global validation MSE: {global_mse:.6f}")
        print(f"Global validation R^2: {global_r2:.6f}")
        print(f"Iteration duration: {iter_duration:.2f} seconds")

        # Simple ETA logging
        avg_time = total_time / (step + 1)
        remaining_iters = n_echo - (step + 1)
        if remaining_iters > 0:
            eta_seconds = avg_time * remaining_iters
            print(
                f"Estimated remaining time: {eta_seconds:.1f} seconds "
                f"(avg {avg_time:.1f} s/iteration over {step + 1} iterations)"
            )

        # 3) Update best-so-far
        if global_mse < best_global_mse:
            print(">>> New best LSTM model found!")
            best_global_mse = global_mse
            best_model = model
            best_params = params_next

        # 4) Store this run
        run_info = {
            "iteration_id": iteration_id,
            "strategy": "bayes",
            "params": params_next,
            "model": model,
            "global_mse": global_mse,
            "global_r2": global_r2,
            "per_horizon_metrics": per_horizon_df,
            "varianceofMSE_across_horizons": variance_per_horizon(per_horizon_df),
        }
        all_results.append(run_info)

        # 5) Feed new observation back into optimizer
        optimizer.tell(x_next, global_mse)

    print("\n=== Bayesian optimization for LSTM finished ===")
    print(f"Best global MSE after BO: {best_global_mse:.6f}")
    print("Best params:", best_params)

    return best_model, best_params, best_global_mse, all_results

#----GRU model functions -----

def build_gru_model(
    input_shape: Tuple[int, int],
    output_dim: int,
    params: Dict
) -> tf.keras.Model:
    """
    Build a multi-output GRU model (sequence -> vector).

    Parameters
    ----------
    input_shape : tuple
        (timesteps, n_features)
    output_dim : int
        Number of outputs (e.g. forecast horizon, 72)
    params : dict
        Hyperparameters for the model. Expected keys (with defaults):
        - units: int
        - num_layers: int (1 or 2 typical)
        - dropout: float
        - recurrent_dropout: float
        - learning_rate: float

    Returns
    -------
    model : tf.keras.Model
        Compiled GRU model ready for training.
    """
    units = params.get("units", 64)
    num_layers = params.get("num_layers", 1)
    dropout = params.get("dropout", 0.0)
    recurrent_dropout = params.get("recurrent_dropout", 0.0)
    learning_rate = params.get("learning_rate", 1e-3)

    model = Sequential()

    # First / only GRU layer
    if num_layers == 1:
        model.add(
            GRU(
                units,
                input_shape=input_shape,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
    else:
        # First GRU with return_sequences=True
        model.add(
            GRU(
                units,
                input_shape=input_shape,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
        # Second / last GRU
        model.add(
            GRU(
                units,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )

    # Output layer: forecast horizon (e.g. 72 hours)
    model.add(Dense(output_dim))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

    return model


def find_best_GRU_model(
    X_train,
    y_train,
    X_val,
    y_val,
    param_scope,
    n_random_start=5,
    n_echo=20,
    random_state=0,
):
    """
    Random + Bayesian hyperparameter search for the GRU model.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs, shape (n_samples, timesteps, n_features).
    y_train : np.ndarray
        Training targets, shape (n_samples, horizon).
    X_val : np.ndarray
        Validation inputs, shape (n_val_samples, timesteps, n_features).
    y_val : np.ndarray
        Validation targets, shape (n_val_samples, horizon).
    param_scope : dict
        Hyperparameter search space for GRU, same structure as for RF/XGB:
        {
            "units": {"type": "int", "bounds": (8, 128)},
            ...
        }
    n_random_start : int
        Number of pure random search iterations before Bayesian phase.
    n_echo : int
        Number of Bayesian (BO) iterations.
    random_state : int
        Random seed.

    Returns
    -------
    best_model : tf.keras.Model
    best_params : dict
    best_global_mse : float
    all_results : list of dict
    results_df : pandas.DataFrame
    per_horizon_df : pandas.DataFrame
    """

    # ------------------------------------------------------------------
    # Logging / call context (same style as RF / XGB)
    # ------------------------------------------------------------------
    call_id = uuid.uuid4().hex[:8]

    stack = inspect.stack()
    caller = stack[1]  # caller frame
    caller_file = caller.filename
    caller_line = caller.lineno
    caller_function = caller.function

    print(f"\n==== NEW CALL TO find_best_GRU_model ====")
    print(f"call_id={call_id}, pid={os.getpid()}")
    print(f"Called from: file='{caller_file}', line={caller_line}, function='{caller_function}'\n")

    # ------------------------------------------------------------------
    # State containers
    # ------------------------------------------------------------------
    all_results = []
    best_model = None
    best_params = None
    best_global_mse = +np.inf

    # For building models
    input_shape = X_train.shape[1:]   # (timesteps, n_features)
    output_dim = y_train.shape[1]     # horizon

    # ------------------------------------------------------------------
    # Helper to sample random hyperparameters from param_scope
    # ------------------------------------------------------------------
    def sample_params(param_scope: Dict[str, Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
        """
        Sample a random hyperparameter configuration from the given scope.
        Supports types: "int", "float", "categorical".
        If "prior" == "log-uniform" for floats, sample in log-space.
        """
        params = {}
        for name, spec in param_scope.items():
            ptype = spec.get("type", "float")
            if ptype == "int":
                low, high = spec["bounds"]
                # rng.integers is [low, high), so we add +1 to include high
                val = int(rng.integers(low, high + 1))
            elif ptype == "float":
                low, high = spec["bounds"]
                prior = spec.get("prior", "uniform")
                if prior == "log-uniform":
                    log_low, log_high = np.log(low), np.log(high)
                    val = float(np.exp(rng.uniform(log_low, log_high)))
                else:
                    val = float(rng.uniform(low, high))
            elif ptype == "categorical":
                values = spec["values"]
                idx = rng.integers(0, len(values))
                val = values[idx]
            else:
                raise ValueError(f"Unsupported param type '{ptype}' for '{name}'")

            params[name] = val
        return params

    # ------------------------------------------------------------------
    # Random search phase
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=random_state)

    print("=== Starting GRU random search phase ===")
    for i in range(n_random_start):
        print(f"\n--- Random Search iteration {i+1}/{n_random_start} ---")

        params = sample_params(param_scope, rng)
        print("Sampled params:", params)

        # Ensure types are consistent
        units = int(params.get("units", 64))
        num_layers = int(params.get("num_layers", 1))
        dropout = float(params.get("dropout", 0.1))
        recurrent_dropout = float(params.get("recurrent_dropout", 0.1))
        batch_size = int(params.get("batch_size", 64))
        epochs = int(params.get("epochs", 50))
        patience = int(params.get("patience", 5))
        learning_rate = float(params.get("learning_rate", 1e-3))

        # Rebuild a params dict that we pass into build_gru_model
        model_params = {
            "units": units,
            "num_layers": num_layers,
            "dropout": dropout,
            "recurrent_dropout": recurrent_dropout,
            "learning_rate": learning_rate,
        }

        # Build GRU model
        model = build_gru_model(input_shape, output_dim, model_params)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )

        # Train
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Predict and evaluate
        y_val_pred = model.predict(X_val, verbose=0)

        global_mse = mean_squared_error(y_val, y_val_pred)
        global_r2 = r2_score(y_val.reshape(-1), y_val_pred.reshape(-1))

        per_horizon_df = evaluate_per_horizon(y_val, y_val_pred)
        var_mse = variance_per_horizon(per_horizon_df)

        print(f"Global validation MSE: {global_mse:.6f}")
        print(f"Global validation R^2: {global_r2:.6f}")
        print(f"Variance of MSE across horizons: {var_mse}")

        trial_result = {
            "iteration_id": i + 1,
            "strategy": "random",
            "params": params,   # keep original param dict including batch_size, epochs, patience
            "model": model,
            "global_mse": float(global_mse),
            "global_r2": global_r2,
            "per_horizon_metrics": per_horizon_df,
            "varianceofMSE_across_horizons": var_mse,
        }
        all_results.append(trial_result)

        if global_mse < best_global_mse:
            best_global_mse = global_mse
            best_model = model
            best_params = params

    # ------------------------------------------------------------------
    # Bayesian optimization phase (to be implemented separately)
    # ------------------------------------------------------------------
    print("\n>>> Starting Bayesian Optimization Phase (GRU) <<<")
    best_model, best_params, best_global_mse, all_results = GRU_baysian_hyperparam_optimization(
        best_model,
        best_params,
        best_global_mse,
        all_results,
        X_train,
        y_train,
        X_val,
        y_val,
        param_scope,
        n_echo,
        n_random_start,
    )

    # ------------------------------------------------------------------
    # Summarize all results
    # ------------------------------------------------------------------
    results_df, per_horizon_df = summarize_all_results_to_df(
        all_results,
        metric_key="global_mse",
    )

    return best_model, best_params, best_global_mse, all_results, results_df, per_horizon_df


def GRU_baysian_hyperparam_optimization(
    best_model,
    best_params,
    best_global_mse,
    all_results,
    X_train,
    y_train,
    X_val,
    y_val,
    param_scope,
    n_echo,
    n_random_start,
):
    """
    Run Bayesian optimization (with skopt.Optimizer) for the GRU model,
    starting from past runs stored in all_results.

    This mirrors XGB_baysian_hyperparam_optimization but uses the GRU
    sequence-to-vector network on 3D inputs.
    """
    print(f"[DEBUG GRU BO] New call to GRU_baysian_hyperparam_optimization")
    # ======================================================================
    # 1) Small helpers: dict <-> vector
    # ======================================================================
    def params_dict_to_vector(params: Dict[str, Any], param_names: List[str]) -> List[Any]:
        """Convert a parameter dictionary into a list following param_names order."""
        return [params[name] for name in param_names]

    def vector_to_params_dict(values: List[Any], param_names: List[str]) -> Dict[str, Any]:
        """Convert a vector/list back into a param dictionary."""
        return {name: v for name, v in zip(param_names, values)}

    # ======================================================================
    # 2) Extract past X, y from all_results
    # ======================================================================
    def extract_Xy_from_results(
        all_results: List[Dict[str, Any]],
        param_names: List[str],
        metric_key: str = "global_mse",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build X_seen and y_seen from past results for BO initialization."""
        X_list = []
        y_list = []

        for res in all_results:
            if "params" not in res:
                continue
            if metric_key not in res:
                continue
            params = res["params"]
            # Keep only runs that have all required params
            if all(name in params for name in param_names):
                x_vec = params_dict_to_vector(params, param_names)
                metric_value = res[metric_key]
                X_list.append(x_vec)
                y_list.append(metric_value)

        if len(X_list) == 0:
            # No past observations
            return np.empty((0, len(param_names))), np.empty((0,))

        X_seen = np.array(X_list)
        y_seen = np.array(y_list, dtype=float)
        return X_seen, y_seen

    # ======================================================================
    # 3) Build a skopt.Optimizer from past results
    # ======================================================================
    def create_bayes_optimizer_from_results(
        all_results: List[Dict[str, Any]],
        param_scope: Dict[str, Dict[str, Any]],
        metric_key: str = "global_mse",
        random_state: int = 0,
        base_estimator: str = "GP",
        acq_func: str = "EI",
    ) -> Tuple["Optimizer", List[str], np.ndarray, np.ndarray]:
        """Prepare the skopt.Optimizer and feed it with past observations."""
        # 1. Build skopt dimensions
        param_names: List[str] = list(param_scope.keys())
        dimensions = []
        for name in param_names:
            spec = param_scope[name]
            ptype = spec.get("type", "float")
            if ptype == "int":
                low, high = spec["bounds"]
                dimensions.append(Integer(low, high, name=name))
            elif ptype == "float":
                low, high = spec["bounds"]
                prior = spec.get("prior", "uniform")
                if prior == "log-uniform":
                    dimensions.append(Real(low, high, prior="log-uniform", name=name))
                else:
                    dimensions.append(Real(low, high, prior="uniform", name=name))
            elif ptype == "categorical":
                values = spec["values"]
                dimensions.append(Categorical(values, name=name))
            else:
                raise ValueError(f"Unsupported parameter type '{ptype}' for '{name}'")

        # 2. Extract past observations
        X_seen, y_seen = extract_Xy_from_results(
            all_results=all_results,
            param_names=param_names,
            metric_key=metric_key,
        )

        # 3. Create the optimizer
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=base_estimator,
            acq_func=acq_func,
            random_state=random_state,
            n_initial_points=0,  # we already have initial points from random search
        )

        # Feed past observations to optimizer (if any)
        if X_seen.shape[0] > 0:
            for x_vec, y_val in zip(X_seen, y_seen):
                optimizer.tell(list(x_vec), float(y_val))

        return optimizer, param_names, X_seen, y_seen

    # ======================================================================
    # 4) Train & evaluate ONE GRU config (used inside BO loop)
    # ======================================================================
    def _train_and_evaluate_gru_for_bayes(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
    ):
        """Train a GRU model with given params and compute validation metrics."""
        # Coerce values and extract training hyperparameters
        units = int(params.get("units", 64))
        num_layers = int(params.get("num_layers", 1))
        dropout = float(params.get("dropout", 0.1))
        recurrent_dropout = float(params.get("recurrent_dropout", 0.1))
        batch_size = int(params.get("batch_size", 64))
        epochs = int(params.get("epochs", 50))
        patience = int(params.get("patience", 5))
        learning_rate = float(params.get("learning_rate", 1e-3))

        # Input/output shapes (3D -> 2D)
        input_shape = X_train.shape[1:]   # (timesteps, n_features)
        output_dim = y_train.shape[1]     # horizon size

        # Params for the model-building function
        model_params = {
            "units": units,
            "num_layers": num_layers,
            "dropout": dropout,
            "recurrent_dropout": recurrent_dropout,
            "learning_rate": learning_rate,
        }

        model = build_gru_model(input_shape, output_dim, model_params)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        y_val_pred = model.predict(X_val, verbose=0)

        global_mse = mean_squared_error(y_val, y_val_pred)
        global_r2 = r2_score(y_val.reshape(-1), y_val_pred.reshape(-1))
        per_horizon_df = evaluate_per_horizon(y_val, y_val_pred)

        return model, float(global_mse), float(global_r2), per_horizon_df

    # ======================================================================
    # 5) Main Bayesian loop
    # ======================================================================
    metric_key = "global_mse"
    random_state = 0  # consistent with other models

    if all_results is None:
        all_results = []

    optimizer, param_names, X_seen, y_seen = create_bayes_optimizer_from_results(
        all_results=all_results,
        param_scope=param_scope,
        metric_key=metric_key,
        random_state=random_state,
    )

    already_evaluated = X_seen.shape[0]
    total_planned = already_evaluated + n_echo

    print("\n=== Starting Bayesian optimization loop (GRU) ===")
    print(f"Already evaluated models (random + previous BO): {already_evaluated}")
    print(f"Will run {n_echo} new Bayesian iterations.")
    print(f"Iteration indexing will start at {already_evaluated + 1} and go up to {total_planned}.")

    total_time = 0.0
    current_iter_index = already_evaluated

    for step in range(n_echo):
        iteration_id = current_iter_index + step + 1
        print(f"\n--- Bayesian iteration {iteration_id}/{total_planned} ---")

        # 1) Ask optimizer for a new candidate
        x_next = optimizer.ask()
        params_next = vector_to_params_dict(x_next, param_names)
        print("Proposed hyperparameters:", params_next)

        # 2) Train & evaluate this GRU config
        iter_start_time = time.time()
        model, global_mse, global_r2, per_horizon_df = _train_and_evaluate_gru_for_bayes(
            X_train,
            y_train,
            X_val,
            y_val,
            params_next,
        )
        iter_duration = time.time() - iter_start_time
        total_time += iter_duration

        print(f"Global validation MSE: {global_mse:.6f}")
        print(f"Global validation R^2: {global_r2:.6f}")
        print(f"Iteration duration: {iter_duration:.2f} seconds")

        # Simple ETA logging
        avg_time = total_time / (step + 1)
        remaining_iters = n_echo - (step + 1)
        if remaining_iters > 0:
            eta_seconds = avg_time * remaining_iters
            print(
                f"Estimated remaining time: {eta_seconds:.1f} seconds "
                f"(avg {avg_time:.1f} s/iteration over {step + 1} iterations)"
            )


        # 3) Update best-so-far
        if global_mse < best_global_mse:
            print(">>> New best GRU model found!")
            best_global_mse = global_mse
            best_model = model
            best_params = params_next

        # 4) Store this run
        run_info = {
            "iteration_id": iteration_id,
            "strategy": "bayes",
            "params": params_next,
            "model": model,
            "global_mse": global_mse,
            "global_r2": global_r2,
            "per_horizon_metrics": per_horizon_df,
            "varianceofMSE_across_horizons": variance_per_horizon(per_horizon_df),
        }
        all_results.append(run_info)

        # 5) Feed new observation back into optimizer
        optimizer.tell(x_next, global_mse)

    print("\n=== Bayesian optimization complete (GRU) ===")
    print(f"Best global MSE so far: {best_global_mse:.6f}")
    print("Best hyperparameters:", best_params)

    return best_model, best_params, best_global_mse, all_results

#----XGBoost model functions -----

def find_best_XGBoost_model(
    X_train,
    y_train,
    X_val,
    y_val,
    param_scope,
    n_random_start=5,
    n_echo=20,
    random_state=0,
):

    # ------------------------------------------------------------------
    # Logging / call context (same style as Random Forest version)
    # ------------------------------------------------------------------
    call_id = uuid.uuid4().hex[:8]

    stack = inspect.stack()
    caller = stack[1]  # caller frame
    caller_file = caller.filename
    caller_line = caller.lineno
    caller_function = caller.function

    print(f"\n==== NEW CALL TO find_best_XGBoost_model ====")
    print(f"call_id={call_id}, pid={os.getpid()}")
    print(f"Called from: file='{caller_file}', line={caller_line}, function='{caller_function}'\n")

    # ------------------------------------------------------------------
    # State containers
    # ------------------------------------------------------------------
    all_results = []
    best_model = None
    best_params = None
    best_global_mse = +np.inf

    # ------------------------------------------------------------------
    # Helper to sample random hyperparameters from param_scope
    # ------------------------------------------------------------------
    def sample_params(param_scope, rng):
        params = {}
        for name, spec in param_scope.items():
            ptype = spec["type"]
            low, high = spec["bounds"]
            if ptype == "int":
                params[name] = rng.integers(low, high + 1)
            elif ptype == "float":
                params[name] = rng.uniform(low, high)
            else:
                raise ValueError(f"Unsupported param type '{ptype}' for '{name}'")
        return params

    # ------------------------------------------------------------------
    # Random search phase (same structure as RF, but with XGBoost model)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=random_state)

    for i in range(n_random_start):
        print(f"\n--- Random Search iteration {i+1}/{n_random_start} ---")

        # Hyperparameters drawn from the provided search space
        params = sample_params(param_scope, rng)

        # Base params that we always use for XGBoost
        base_params = {
            "objective": "reg:squarederror",  # MSE loss
            "n_jobs": -1,
            "random_state": random_state,
        }
        base_params.update(params)

        # XGBoost base model
        base_model = XGBRegressor(**base_params)

        # Multi-output wrapper (one XGB model per horizon)
        model = MultiOutputRegressor(base_model)

        # Fit + predict
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        # Global metrics
        global_mse = mean_squared_error(y_val, y_val_pred)
        global_r2 = r2_score(y_val.reshape(-1), y_val_pred.reshape(-1))

        # Per-horizon metrics
        per_horizon_df = evaluate_per_horizon(y_val, y_val_pred)

        # Store trial result (same structure as RF)
        trial_result = {
            "iteration_id": i + 1,
            "strategy": "random",
            "params": params,  # only the search-space params, like in RF
            "model": model,
            "global_mse": float(global_mse),
            "global_r2": global_r2,
            "per_horizon_metrics": per_horizon_df,
            "varianceofMSE_across_horizons": variance_per_horizon(per_horizon_df),
        }

        all_results.append(trial_result)

        # Track best model so far
        if global_mse < best_global_mse:
            best_global_mse = global_mse
            best_model = model
            best_params = params

    # ------------------------------------------------------------------
    # Bayesian optimization phase (XGBoost version – to be implemented)
    # ------------------------------------------------------------------
    print("\n>>> Starting Bayesian Optimization Phase (XGBoost) <<<")
    best_model, best_params, best_global_mse, all_results = XGB_baysian_hyperparam_optimization(
        best_model,
        best_params,
        best_global_mse,
        all_results,
        X_train,
        y_train,
        X_val,
        y_val,
        param_scope,
        n_echo,
        n_random_start,
    )

    # ------------------------------------------------------------------
    # Summarize all results (same utilities as RF)
    # ------------------------------------------------------------------
    results_df, per_horizon_df = summarize_all_results_to_df(
        all_results,
        metric_key="global_mse",
    )

    return best_model, best_params, best_global_mse, all_results, results_df, per_horizon_df


def XGB_baysian_hyperparam_optimization(
    best_model,
    best_params,
    best_global_mse,
    all_results,
    X_train,
    y_train,
    X_val,
    y_val,
    param_scope,
    n_echo,
    n_random_start,
):
    """
    Run Bayesian optimization (with skopt.Optimizer) for the XGBoost model,
    starting from past runs stored in all_results.

    This mirrors Tree_baysian_hyperparam_optimization but uses XGBRegressor
    instead of RandomForestRegressor.
    """

    # ======================================================================
    # 1) Helper: build the skopt search space from param_scope
    # ======================================================================
    def build_skopt_search_space(
        param_scope: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[Any]]:
        """
        Convert a parameter scope dict into a list of parameter names and
        a list of skopt Dimension objects (Integer, Real, Categorical).

        Example param_scope:
        {
            "n_estimators":    {"type": "int",   "bounds": (100, 200)},
            "max_depth":       {"type": "int",   "bounds": (3, 5)},
            "learning_rate":   {"type": "float", "bounds": (0.05, 0.2), "prior": "log-uniform"},
            "subsample":       {"type": "float", "bounds": (0.7, 1.0)},
            "colsample_bytree":{"type": "float", "bounds": (0.7, 1.0)},
        }
        """
        param_names: List[str] = []
        dimensions: List[Any] = []

        for name, spec in param_scope.items():
            ptype = spec["type"]

            if ptype == "int":
                low, high = spec["bounds"]
                dim = Integer(low, high, name=name)

            elif ptype == "float":
                low, high = spec["bounds"]
                prior = spec.get("prior", "uniform")
                dim = Real(low, high, prior=prior, name=name)

            elif ptype == "categorical":
                values = spec["values"]
                dim = Categorical(values, name=name)

            else:
                raise ValueError(f"Unsupported param type '{ptype}' for '{name}'")

            param_names.append(name)
            dimensions.append(dim)

        return param_names, dimensions

    # ======================================================================
    # 2) Helpers to convert dict <-> vector for skopt
    # ======================================================================
    def params_dict_to_vector(params: Dict[str, Any], param_names: List[str]) -> List[Any]:
        """
        Convert a parameter dictionary into a list/vector following param_names order.
        """
        return [params[name] for name in param_names]

    def vector_to_params_dict(values: List[Any], param_names: List[str]) -> Dict[str, Any]:
        """
        Convert a vector/list of parameter values back into a dict.
        """
        return {name: v for name, v in zip(param_names, values)}

    # ======================================================================
    # 3) Extract past X, y from all_results
    # ======================================================================
    def extract_Xy_from_results(
        all_results: List[Dict[str, Any]],
        param_names: List[str],
        metric_key: str = "global_mse",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build X_seen and y_seen from past results.

        all_results: list of dict, each with at least
            - "params": dict of hyperparameters
            - metric_key: scalar objective value
        """
        X_list = []
        y_list = []

        for res in all_results:
            if "params" not in res:
                continue
            if metric_key not in res:
                continue

            params = res["params"]
            # Keep only runs that have all required params
            if all(name in params for name in param_names):
                x_vec = params_dict_to_vector(params, param_names)
                metric_value = res[metric_key]
                X_list.append(x_vec)
                y_list.append(metric_value)

        if len(X_list) == 0:
            # No past results: return empty arrays
            return np.empty((0, len(param_names))), np.empty((0,))

        X_seen = np.array(X_list)
        y_seen = np.array(y_list, dtype=float)
        return X_seen, y_seen

    # ======================================================================
    # 4) Build a skopt.Optimizer from past results
    # ======================================================================
    def create_bayes_optimizer_from_results(
        all_results: List[Dict[str, Any]],
        param_scope: Dict[str, Dict[str, Any]],
        metric_key: str = "global_mse",
        random_state: int = 0,
        base_estimator: str = "GP",
        acq_func: str = "EI",
    ) -> Tuple[Optimizer, List[str], np.ndarray, np.ndarray]:
        """
        Prepare everything needed for Bayesian optimization with skopt:

        1. Build the search space (param_names, dimensions).
        2. Extract past (X_seen, y_seen) from all_results.
        3. Instantiate a skopt.Optimizer and feed it the past observations.
        """
        # 1. Build skopt dimensions
        param_names, dimensions = build_skopt_search_space(param_scope)

        # 2. Extract past observations
        X_seen, y_seen = extract_Xy_from_results(
            all_results=all_results,
            param_names=param_names,
            metric_key=metric_key,
        )

        # 3. Create the optimizer
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=base_estimator,
            acq_func=acq_func,
            random_state=random_state,
            n_initial_points=0,  # we already have initial points from X_seen, y_seen
        )

        # Feed past data to the optimizer (if any)
        if X_seen.shape[0] > 0:
            optimizer.tell(X_seen.tolist(), y_seen.tolist())

        return optimizer, param_names, X_seen, y_seen

    # ======================================================================
    # 5) Train & evaluate ONE XGBoost config (used inside BO loop)
    # ======================================================================
    def _train_and_evaluate_xgb_for_bayes(
        X_train,
        y_train,
        X_val,
        y_val,
        params: Dict[str, Any],
        random_state: int = 0,
    ):
        """
        Train a multi-output XGBoost model with given params and compute:
        - global MSE on the validation set (objective for BO)
        - per-horizon metrics
        """
        # Base XGBoost parameters (we always set these)
        base_params = {
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": random_state,
        }
        base_params.update(params)

        # One XGBRegressor per horizon
        base_model = XGBRegressor(**base_params)
        model = MultiOutputRegressor(base_model)

        # Fit and predict
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        # Global metrics
        global_mse = mean_squared_error(y_val, y_val_pred)
        global_r2 = r2_score(y_val.reshape(-1), y_val_pred.reshape(-1))

        # Per-horizon metrics
        per_horizon_df = evaluate_per_horizon(y_val, y_val_pred)

        return model, float(global_mse), float(global_r2), per_horizon_df

    # ======================================================================
    # 6) Main Bayesian loop
    # ======================================================================
    metric_key = "global_mse"
    random_state = 0  # keep same as RF version

    if all_results is None:
        all_results = []

    # ----- Part 1: build optimizer from past results -----
    optimizer, param_names, X_seen, y_seen = create_bayes_optimizer_from_results(
        all_results=all_results,
        param_scope=param_scope,
        metric_key=metric_key,
        random_state=random_state,
    )

    # For iteration numbering
    current_iter_index = n_random_start
    total_planned = n_random_start + n_echo

    print("\n=== Starting Bayesian optimization loop (XGBoost) ===")
    print(f"Already evaluated models (random + previous BO): {X_seen.shape[0]}")
    print(f"Will run {n_echo} new Bayesian iterations.")
    print(f"Iteration indexing will start at {current_iter_index + 1} and go up to {total_planned}.\n")

    total_time = 0.0

    # ----- Part 2: BO iterations -----
    for step in range(n_echo):
        iteration_id = current_iter_index + step + 1
        print(f"\n--- Bayesian iteration {iteration_id}/{total_planned} ---")

        # 2.1 Ask optimizer for next hyperparameters (vector form)
        x_next = optimizer.ask()
        params_next = vector_to_params_dict(x_next, param_names)
        print("Proposed hyperparameters:", params_next)

        # 2.2 Train model + evaluate
        iter_start_time = time.time()
        model, global_mse, global_r2, per_horizon_df = _train_and_evaluate_xgb_for_bayes(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=params_next,
            random_state=random_state,
        )
        iter_duration = time.time() - iter_start_time
        total_time += iter_duration

        print(f"Global validation MSE: {global_mse:.6f}")
        print(f"Time for this iteration: {iter_duration:.1f} seconds")

        # Simple ETA logging
        avg_time = total_time / (step + 1)
        remaining_iters = n_echo - (step + 1)
        if remaining_iters > 0:
            eta_seconds = avg_time * remaining_iters
            print(
                f"Estimated remaining time: {eta_seconds:.1f} seconds "
                f"(avg {avg_time:.1f} s/iteration over {step + 1} iterations)"
            )

        # 2.3 Update best model if improved
        if (best_global_mse is None) or (global_mse < best_global_mse):
            best_global_mse = global_mse
            best_model = model
            best_params = params_next
            print(" --> New best XGBoost model found! 🎉")
        else:
            print("No improvement over current best model.")

        # 2.4 Store results for this run
        run_info = {
            "iteration_id": iteration_id,
            "strategy": "bayes",
            "params": params_next,
            metric_key: global_mse,
            "global_r2": global_r2,
            "per_horizon_metrics": per_horizon_df,
            "varianceofMSE_across_horizons": variance_per_horizon(per_horizon_df),
        }
        all_results.append(run_info)

        # 2.5 Feed new observation back into optimizer
        optimizer.tell(x_next, global_mse)

    print("\n=== Bayesian optimization complete (XGBoost) ===")
    print(f"Best global MSE so far: {best_global_mse:.6f}")
    print("Best hyperparameters:", best_params)

    return best_model, best_params, best_global_mse, all_results

#----Random Forest model functions -----

def find_best_random_forest_model (X_train,y_train,X_val,y_val,param_scope,n_random_start=5,n_echo=20,random_state=0,):

    #defining variables and intern functions
    call_id = uuid.uuid4().hex[:8]

    # Get caller information
    stack = inspect.stack()
    caller = stack[1]  # caller frame
    caller_file = caller.filename
    caller_line = caller.lineno
    caller_function = caller.function

    print(f"\n==== NEW CALL TO find_best_random_forest_model ====")
    print(f"call_id={call_id}, pid={os.getpid()}")
    print(f"Called from: file='{caller_file}', line={caller_line}, function='{caller_function}'\n")


    all_results = []        
    best_model = None
    best_params = None
    best_global_mse = +np.inf

    #select a set of random hyperparameters
    def sample_params(param_scope, rng):
        params = {}
        for name, spec in param_scope.items():
            ptype = spec["type"]
            low, high = spec["bounds"]
            if ptype == "int":
                params[name] = rng.integers(low, high + 1)
            elif ptype == "float":
                params[name] = rng.uniform(low, high)
            else:
                raise ValueError("Unsupported param type")
        return params

    
    #evaluate the first parameters
    rng = np.random.default_rng(seed=random_state)
    for i in range(n_random_start):
        print(f"\n--- Random Search iteration {i+1}/{n_random_start} ---")
        params = sample_params(param_scope, rng)
        
        model = RandomForestRegressor(
            **params,
            n_jobs=-1,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        global_mse = mean_squared_error(y_val, y_val_pred)
        global_r2 = r2_score(y_val.reshape(-1), y_val_pred.reshape(-1))

        #per-horizon metrics as a DataFrame (same as BO)
        per_horizon_df = evaluate_per_horizon(y_val, y_val_pred)


        trial_result = {
        "iteration_id": i+1,
        "strategy": "random",
        "params": params,
        "model": model,
        "global_mse": float(global_mse),
        "global_r2": global_r2,
        "per_horizon_metrics": per_horizon_df,   # or a DataFrame
        "varianceofMSE_across_horizons": variance_per_horizon(per_horizon_df)
        }
        
        all_results.append(trial_result)


        if global_mse < best_global_mse:
            best_global_mse = global_mse
            best_model = model
            best_params = params


    print("\n>>> Starting Bayesian Optimization Phase <<<")
    best_model, best_params, best_global_mse, all_results = Tree_baysian_hyperparam_optimization(
                                                            best_model, best_params, 
                                                            best_global_mse, all_results,
                                                            X_train,y_train,X_val,y_val,
                                                            param_scope, n_echo, n_random_start
                                                            )


    results_df, per_horizon_df = summarize_all_results_to_df(all_results, metric_key="global_mse")
    

    return best_model, best_params, best_global_mse, all_results, results_df, per_horizon_df


def Tree_baysian_hyperparam_optimization(best_model, best_params, 
                                        best_global_mse, all_results,
                                        X_train,y_train,X_val,y_val,
                                        param_scope, n_echo, n_random_start):

    #define function to prepare the data

    def build_skopt_search_space(
                                    param_scope: Dict[str, Dict[str, Any]]
                                    ) -> Tuple[List[str], List[Any]]:
        """
        Convert a parameter scope dict into a list of parameter names and
        a list of skopt Dimension objects (Integer, Real, Categorical).

        Parameters
        ----------
        param_scope : dict
            Dictionary describing the hyperparameter search space.
            Example:
            {
                "n_estimators": {"type": "int", "bounds": (50, 300)},
                "max_depth": {"type": "int", "bounds": (5, 30)},
                "min_samples_leaf": {"type": "int", "bounds": (1, 10)},
                "learning_rate": {"type": "float", "bounds": (0.01, 0.3), "prior": "log-uniform"},
                "criterion": {"type": "categorical", "values": ["squared_error", "absolute_error"]},
            }

        Returns
        -------
        param_names : list of str
            Names of the parameters, in a fixed order.
        dimensions : list of skopt.space.Dimension
            Dimension objects compatible with skopt.Optimizer.
        """
        param_names: List[str] = []
        dimensions: List[Any] = []

        for name, spec in param_scope.items():
            p_type = spec.get("type")
            if p_type is None:
                raise ValueError(f"Parameter '{name}' is missing a 'type' field.")

            if p_type == "int":
                low, high = spec["bounds"]
                dim = Integer(low, high, name=name)

            elif p_type == "float":
                low, high = spec["bounds"]
                prior = spec.get("prior", "uniform")  # e.g. "uniform" or "log-uniform"
                dim = Real(low, high, prior=prior, name=name)

            elif p_type == "categorical":
                values = spec["values"]
                dim = Categorical(values, name=name)

            else:
                raise ValueError(f"Unsupported parameter type '{p_type}' for '{name}'.")

            param_names.append(name)
            dimensions.append(dim)

        return param_names, dimensions
    
    def params_dict_to_vector(params: Dict[str, Any], param_names: List[str]) -> List[Any]:
        """
        Convert a parameter dictionary into a list/vector following param_names order.

        Parameters
        ----------
        params : dict
            Hyperparameter dictionary, e.g. {"n_estimators": 100, "max_depth": 5, ...}
        param_names : list of str
            The order of parameter names that defines the vector layout.

        Returns
        -------
        vector : list
            Parameter values in the same order as param_names.
        """
        return [params[name] for name in param_names]

    def vector_to_params_dict(values: List[Any], param_names: List[str]) -> Dict[str, Any]:
        """
        Convert a vector/list of parameter values back into a dict.

        Parameters
        ----------
        values : list
            Parameter values in the same order as param_names.
        param_names : list of str
            Names of the parameters.

        Returns
        -------
        params : dict
            Dictionary mapping parameter names to values.
        """
        return {name: value for name, value in zip(param_names, values)}

    def extract_Xy_from_results(
        all_results: List[Dict[str, Any]],
        param_names: List[str],
        metric_key: str = "global_mse"
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build X_seen and y_seen from past results.

        Parameters
        ----------
        all_results : list of dict
            Past model evaluations. Each element should at least contain:
            - "params": dict of hyperparameters
            - metric_key: scalar objective value (e.g. "global_mse")
        Example of one entry:
            {
                "iteration_id": 3,
                "strategy": "random",
                "params": {...},
                "global_mse": 0.123,
                "per_horizon_metrics": ...
            }
        param_names : list of str
            Order of parameters used to build vectors.
        metric_key : str, optional (default "global_mse")
            Key in each result dict where the objective value is stored.

        Returns
        -------
        X_seen : np.ndarray, shape (n_runs, n_params)
            Matrix of past hyperparameter vectors.
        y_seen : np.ndarray, shape (n_runs,)
            Vector of past objective values.
        """
        X_list = []
        y_list = []

        for res in all_results:
            if "params" not in res:
                continue
            if metric_key not in res:
                continue

            params = res["params"]
            metric_value = res[metric_key]

            x_vec = params_dict_to_vector(params, param_names)
            X_list.append(x_vec)
            y_list.append(metric_value)

        if len(X_list) == 0:
            # No past results: return empty arrays
            return np.empty((0, len(param_names))), np.empty((0,))

        X_seen = np.array(X_list)
        y_seen = np.array(y_list, dtype=float)
        return X_seen, y_seen

    def create_bayes_optimizer_from_results(
        all_results: List[Dict[str, Any]],
        param_scope: Dict[str, Dict[str, Any]],
        metric_key: str = "global_mse",
        random_state: int = 0,
        base_estimator: str = "GP",
        acq_func: str = "EI"
    ) -> Tuple[Optimizer, List[str], np.ndarray, np.ndarray]:
        """
        Prepare everything needed for Bayesian optimization with skopt:

        1. Build the search space (param_names, dimensions).
        2. Extract past (X_seen, y_seen) from all_results.
        3. Instantiate a skopt.Optimizer and feed it the past observations.

        Parameters
        ----------
        all_results : list of dict
            Past model evaluations (random search + maybe previous BO runs).
        param_scope : dict
            Search space definition (see build_skopt_search_space docstring).
        metric_key : str, optional (default "global_mse")
            Name of the metric used as objective (lower is better).
        random_state : int, optional (default 0)
            Random seed for the optimizer.
        base_estimator : str, optional (default "GP")
            Surrogate model type for skopt.Optimizer ("GP", "RF", "ET", "GBRT").
        acq_func : str, optional (default "EI")
            Acquisition function ("EI", "PI", "LCB", ...).

        Returns
        -------
        optimizer : skopt.Optimizer
            Optimizer already 'told' about the past runs.
        param_names : list of str
            Parameter names in the same order as optimizer.space.dimensions.
        X_seen : np.ndarray
            Past hyperparameter vectors (possibly empty).
        y_seen : np.ndarray
            Past objective values (possibly empty).
        """
        # 1. Build skopt dimensions from param_scope
        param_names, dimensions = build_skopt_search_space(param_scope)

        # 2. Extract past observations
        X_seen, y_seen = extract_Xy_from_results(
            all_results=all_results,
            param_names=param_names,
            metric_key=metric_key,
        )

        # 3. Create the optimizer
        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=base_estimator,
            acq_func=acq_func,
            random_state=random_state,
            n_initial_points=0  # we already have initial points from X_seen, y_seen
        )

        # If we have past data, feed it to the optimizer
        if X_seen.shape[0] > 0:
            optimizer.tell(X_seen.tolist(), y_seen.tolist())

        return optimizer, param_names, X_seen, y_seen


    #There we define running the forest and optimize the baysian
    def _train_and_evaluate_rf_for_bayes(
        X_train,
        y_train,
        X_val,
        y_val,
        params: Dict[str, Any],
        random_state: int = 0,
    ):
        """
        Train a multi-output Random Forest with given params and compute:
        - global MSE on the validation set (objective for BO)
        - per-horizon metrics (using evaluate_per_horizon)

        Parameters
        ----------
        X_train, y_train, X_val, y_val : np.ndarray
            Data matrices for training and validation.
            y_* shape should be (n_samples, horizon).
        params : dict
            Hyperparameters for RandomForestRegressor, e.g.:
            {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 2,
            }
        random_state : int
            Random seed.

        Returns
        -------
        model : MultiOutputRegressor
            Trained model.
        global_mse : float
            MSE over all horizons and all samples (flattened).
        per_horizon_df : pandas.DataFrame
            Output of evaluate_per_horizon(y_val, y_val_pred).
        """
        # Safely extract / cast parameters
        n_estimators = int(params.get("n_estimators", 100))
        max_depth_raw = params.get("max_depth", None)
        max_depth = None if max_depth_raw in (None, "None") else int(max_depth_raw)
        min_samples_leaf = int(params.get("min_samples_leaf", 1))

        base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base_rf)

        # Train
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_val_pred = model.predict(X_val)

        # Global MSE (objective for BO)
        global_mse = mean_squared_error(
            y_val.reshape(-1),
            y_val_pred.reshape(-1)
        )

        #Global R2
        global_r2 = r2_score(y_val.reshape(-1), y_val_pred.reshape(-1))
        
        # Per-horizon metrics for analysis
        per_horizon_df = evaluate_per_horizon(y_val, y_val_pred)

        return model, global_mse,global_r2, per_horizon_df

    #--- Main Bayesian Optimization Loop ---
    def run_the_baysian_opt_loop(
        best_model,
        best_params,
        best_global_mse,
        all_results,
        X_train,
        y_train,
        X_val,
        y_val,
        param_scope: Dict[str, Dict[str, Any]],
        n_echo: int,
        n_random_start: int,
        metric_key: str = "global_mse",
        random_state: int = 0,
    ):
        """
        Continue hyperparameter search with Bayesian optimization (skopt),
        starting from previous random-search results.

        Parameters
        ----------
        best_model :
            Current best model (from random search and/or previous BO steps).
        best_params : dict
            Hyperparameters of the current best model.
        best_global_mse : float
            Global MSE of the current best model (lower is better).
        all_results : list of dict
            List of all past runs. Each entry should at least contain:
                - "params": dict of hyperparameters
                - metric_key (e.g. "global_mse"): float
            and can contain:
                - "iteration_id"
                - "strategy" ("random" or "bayes")
                - "per_horizon_metrics"
        X_train, y_train, X_val, y_val : np.ndarray
            Training and validation sets.
        param_scope : dict
            Search space definition, e.g.:
            {
                "n_estimators":     {"type": "int",   "bounds": (50, 300)},
                "max_depth":        {"type": "int",   "bounds": (5, 30)},
                "min_samples_leaf": {"type": "int",   "bounds": (1, 10)},
            }
        n_echo : int
            Number of Bayesian iterations to perform in this call.
        n_random_start : int
            Number of models already evaluated before BO (random search).
            Used only for iteration numbering.
        metric_key : str, default "global_mse"
            Name of the metric in all_results used as the BO objective.
        random_state : int, default 0
            Random seed for the skopt optimizer and RF training.

        Returns
        -------
        best_model, best_params, best_global_mse, all_results
            Updated after running n_echo Bayesian iterations.
        """
        if all_results is None:
            all_results = []

        # ===== Part 1: build optimizer from past results =====
        optimizer, param_names, X_seen, y_seen = create_bayes_optimizer_from_results(
            all_results=all_results,
            param_scope=param_scope,
            metric_key=metric_key,
            random_state=random_state,
        )

        # For logging / iteration indexing
        current_iter_index = n_random_start
        total_planned = n_random_start + n_echo

        print("\n=== Starting Bayesian optimization loop ===")
        print(f"Already evaluated models (random + previous BO): {X_seen.shape[0]}")
        print(f"Will run {n_echo} new Bayesian iterations.")
        print(f"Iteration indexing will start at {current_iter_index + 1} and go up to {total_planned}.\n")

        total_time = 0.0
        
        # ===== Part 2: Bayesian loop =====
        for step in range(n_echo):
            iteration_id = current_iter_index + step + 1
            print(f"\n--- Bayesian iteration {iteration_id}/{total_planned} ---")

            # 2.1 Ask optimizer for next hyperparameters (vector form)
            x_next = optimizer.ask()
            params_next = vector_to_params_dict(x_next, param_names)
            print("Proposed hyperparameters:", params_next)

            # 2.2 Train model + evaluate
            iter_start_time = time.time()
            model, global_mse, global_r2, per_horizon_df = _train_and_evaluate_rf_for_bayes(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                params=params_next,
                random_state=random_state,
            )
            iter_duration = time.time() - iter_start_time
            total_time += iter_duration

            print(f"Global validation MSE: {global_mse:.6f}")
            print(f"Time for this iteration: {iter_duration:.1f} seconds")

            # NEW: average time and ETA
            avg_time = total_time / (step + 1)
            remaining_iters = n_echo - (step + 1)
            if remaining_iters > 0:
                eta_seconds = avg_time * remaining_iters
                print(
                    f"Estimated remaining time: {eta_seconds:.1f} seconds "
                    f"(avg {avg_time:.1f} s/iteration over {step + 1} iterations)"
                )


            # Update best model if improved
            if (best_global_mse is None) or (global_mse < best_global_mse):
                best_global_mse = global_mse
                best_model = model
                best_params = params_next
                print(" --> New best model found! 🎉")
            else:
                print("No improvement over current best model.")

            # 2.3 Store results for this run
            run_info = {
                "iteration_id": iteration_id,
                "strategy": "bayes",
                "params": params_next,
                metric_key: global_mse,
                "global_r2": global_r2,
                "per_horizon_metrics": per_horizon_df,
                "varianceofMSE_across_horizons": variance_per_horizon(per_horizon_df)
            }
            all_results.append(run_info)

            # Feed the new observation back into the optimizer
            optimizer.tell(x_next, global_mse)

        print("\n=== Bayesian optimization complete ===")
        print(f"Best global MSE so far: {best_global_mse:.6f}")
        print("Best hyperparameters:", best_params)
       
        return best_model, best_params, best_global_mse, all_results


   
    best_model, best_params, best_global_mse, all_results = run_the_baysian_opt_loop(
                                                                                            best_model,
                                                                                            best_params,
                                                                                            best_global_mse,
                                                                                            all_results,
                                                                                            X_train,
                                                                                            y_train,
                                                                                            X_val,
                                                                                            y_val,
                                                                                            param_scope,
                                                                                            n_echo,
                                                                                            n_random_start,
                                                                                            metric_key="global_mse",
                                                                                            random_state=0,)
                                                                                            
    return best_model, best_params, best_global_mse, all_results


#############################################################################################################
# Show and treat the results of the ML models
#############################################################################################################

def get_best_run_from_results(
    all_results,
    metric_key: str = "global_mse"
):
    """
    Find the best run in all_results according to the given metric.

    Parameters
    ----------
    all_results : list of dict
        Same structure as for summarize_all_results_to_df.
    metric_key : str, default "global_mse"
        Metric to minimize.

    Returns
    -------
    best_run : dict or None
        The best run dictionary, or None if all_results is empty
        or no valid metric is found.
    """
    if not all_results:
        return None

    best_run = None
    best_value = None

    for run in all_results:
        if metric_key not in run:
            continue
        value = run[metric_key]
        if value is None:
            continue

        if (best_value is None) or (value < best_value):
            best_value = value
            best_run = run

    return best_run


def evaluate_per_horizon(y_true, y_pred):
    """
    Compute MAE and RMSE for each forecast horizon separately.

    Parameters
    ----------
    y_true : numpy.ndarray
        True targets of shape (n_samples, horizon).
    y_pred : numpy.ndarray
        Predicted targets of shape (n_samples, horizon).

    Returns
    -------
    metrics_df : pandas.DataFrame
        DataFrame with columns:
        - 'horizon': step index (1 = 1-step ahead, etc.)
        - 'MAE'
        - 'RMSE'
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")

    n_samples, horizon = y_true.shape

    maes, rmses, r2s = [], [], []

    for h in range(horizon):
        true_h = y_true[:, h]
        pred_h = y_pred[:, h]

        mae_h = mean_absolute_error(true_h, pred_h)
        rmse_h = np.sqrt(mean_squared_error(true_h, pred_h))
        r2_h = r2_score(true_h, pred_h)

        maes.append(mae_h)
        rmses.append(rmse_h)
        r2s.append(r2_h)

    metrics_df = pd.DataFrame({
        "horizon": np.arange(1, horizon + 1),
        "MAE": maes,
        "RMSE": rmses,
        "R2": r2s,
    })

    return metrics_df


def summarize_all_results_to_df(
    all_results,
    metric_key: str = "global_mse",
    per_horizon_metric_rmse_col: str = "RMSE",
    per_horizon_r2_col: str = "R2",
):
    """
    Convert the list of run dictionaries (all_results) into:
      1) A flat summary DataFrame (one row per run).
      2) A long per-horizon DataFrame (rows = horizon, model; cols = horizon, iteration_id, MSE, R2).

    Each element of all_results is expected to look like:
        {
            "iteration_id": 3,
            "strategy": "bayes",
            "params": {...},
            "global_mse": 0.123,
            "global_r2": 0.87,                      # optional
            "per_horizon_metrics": <DataFrame>,     # columns: horizon, RMSE, (optional) R2, ...
            "varianceofMSE_across_horizons": 0.0012 # optional
        }
    """
    if not all_results:
        return pd.DataFrame(), pd.DataFrame()

    # ------------------------------------------------------------------
    # 1) Build the flat summary DataFrame 
    # ------------------------------------------------------------------
    # collect all parameter names
    param_keys = sorted({
        key
        for run in all_results
        for key in run.get("params", {}).keys()
    })

    summary_rows = []
    for run in all_results:
        row = {
            "iteration_id": run.get("iteration_id"),
            "strategy": run.get("strategy"),
            metric_key: run.get(metric_key, None),
            # optional extra metrics if present
            "global_r2": run.get("global_r2"),
            "var_mse_across_horizons": run.get("varianceofMSE_across_horizons"),
        }

        params = run.get("params", {})
        for p in param_keys:
            row[p] = params.get(p, None)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if "iteration_id" in summary_df.columns:
        summary_df = summary_df.sort_values("iteration_id").reset_index(drop=True)

    # mark best row by metric_key
    if metric_key in summary_df.columns and summary_df[metric_key].notna().any():
        best_idx = summary_df[metric_key].idxmin()
        summary_df["is_best"] = False
        summary_df.loc[best_idx, "is_best"] = True
    else:
        summary_df["is_best"] = False

    # ------------------------------------------------------------------
    # 2) Build the per-horizon LONG DataFrame:
    #    columns = [horizon, iteration_id, MSE, R2]
    # ------------------------------------------------------------------
    per_horizon_rows = []

    for run in all_results:
        iter_id = run.get("iteration_id")
        per_h_df = run.get("per_horizon_metrics", None)

        if iter_id is None or per_h_df is None:
            continue

        df = per_h_df.copy()

        # we expect a "horizon" column
        if "horizon" not in df.columns:
            continue

        # ensure we have MSE: either directly or from RMSE
        if "MSE" in df.columns:
            df["MSE"] = df["MSE"].astype(float)
        elif per_horizon_metric_rmse_col in df.columns:
            df["MSE"] = (df[per_horizon_metric_rmse_col].astype(float) ** 2)
        else:
            # can't compute MSE -> skip this run for per-horizon export
            continue

        # handle R2 if present
        if per_horizon_r2_col in df.columns:
            df["R2"] = df[per_horizon_r2_col].astype(float)
        else:
            df["R2"] = np.nan  # or leave it out if you prefer

        df["horizon"] = df["horizon"].astype(int)
        df["iteration_id"] = iter_id

        per_horizon_rows.append(df[["horizon", "iteration_id", "MSE", "R2"]])

    if per_horizon_rows:
        per_horizon_df = pd.concat(per_horizon_rows, ignore_index=True)
        per_horizon_df = per_horizon_df.sort_values(["iteration_id", "horizon"]).reset_index(drop=True)
    else:
        per_horizon_df = pd.DataFrame(columns=["horizon", "iteration_id", "MSE", "R2"])

    return summary_df, per_horizon_df


def variance_per_horizon(per_horizon_df):
    """
    Compute the variance of the MSE across horizons, given a
    per_horizon_metrics DataFrame produced by `evaluate_per_horizon`.

    Parameters
    ----------
    per_horizon_df : pandas.DataFrame
        Expected to have at least:
            - 'horizon' column
            - either 'MSE' or 'RMSE' column
        (in our current setup, we have 'RMSE'.)

    Returns
    -------
    var_mse : float or None
        Variance of the MSE across horizons. Returns None if
        no valid MSE values are available.
    """
    if per_horizon_df is None or len(per_horizon_df) == 0:
        return None

    # Try to get MSE directly if it exists, otherwise derive it from RMSE
    if "MSE" in per_horizon_df.columns:
        mse_values = per_horizon_df["MSE"].to_numpy(dtype=float)
    elif "RMSE" in per_horizon_df.columns:
        rmse_values = per_horizon_df["RMSE"].to_numpy(dtype=float)
        mse_values = rmse_values ** 2
    else:
        raise ValueError(
            "per_horizon_df must contain either an 'MSE' or 'RMSE' column."
        )

    # Drop NaNs just in case
    mse_values = mse_values[~np.isnan(mse_values)]
    if mse_values.size == 0:
        return None

    # Variance of MSE across horizons (population variance)
    var_mse = float(np.var(mse_values))

    return var_mse

