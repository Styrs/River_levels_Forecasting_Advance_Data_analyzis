
import pandas as pd
import data_clearing_function
import ML_function
import inspect, uuid, os



def run_XGBoost_model(Config, path_to_csv,lookback_size=72,horizon_size=72):

    run_id = uuid.uuid4().hex[:8]

    stack = inspect.stack()
    caller = stack[1]
    caller_file = caller.filename
    caller_line = caller.lineno
    caller_function = caller.function

    print(f"\n==== NEW CALL TO run_XGB_model ====")
    print(f"run_id={run_id}, pid={os.getpid()}")
    print(f"Called from: file='{caller_file}', line={caller_line}, function='{caller_function}'\n")
    Config = Config   # or "B", "C", "D" #Define what condition we use 
    path_to_csv = path_to_csv


    #############################################################################
    #Data import and final preparation
    #############################################################################


    data_station = ML_function.prepare_station_dataframe(path_to_csv, Config)

    #############################################################################
    #Run the XBG model
    #############################################################################

    lookback_size = 72  # e.g., past 72 hours
    horizon_size = 72   # e.g., next 72 hours

    x_input, y_output = ML_function.make_windowed_data(data_station, target_col="self_data", lookback=lookback_size, horizon=horizon_size)


    x_train, y_train, x_val, y_val, x_test, y_test = ML_function.split_time_series(x_input, y_output, train_frac=0.7, val_frac=0.2)

    print("Start training the XGB model for one station")

    
    param_scope = {
        "n_estimators":   {"type": "int",   "bounds": (1, 150)},
        "max_depth":      {"type": "int",   "bounds": (1, 6)},
        "learning_rate":  {"type": "float", "bounds": (0.05, 0.2), "prior": "log-uniform"},
        "subsample":      {"type": "float", "bounds": (0.7, 1.0)},
        "colsample_bytree":{"type": "float","bounds": (0.7, 1.0)},
    }

    best_model_XGB, best_params_XGB, best_global_mse_XGB, all_results_XGB, results_df_XGB, per_horizon_df_XGB = ML_function.find_best_XGBoost_model(x_train, y_train, x_val, y_val,
                                                                                                param_scope,n_random_start=5,n_echo=50,random_state=0)


    print(best_model_XGB)

    results_df_XGB.to_csv("results_XGB.csv", index=False)
    per_horizon_df_XGB.to_csv("per_horizon_XGB.csv", index=False)


    # ============================================
    # Evaluate BEST XGBoost model on TEST SET
    # ============================================
    print("\nEvaluating best XGBoost model on TEST SET...")

    y_test_pred = best_model_XGB.predict(x_test)

    from sklearn.metrics import mean_squared_error, r2_score

    test_global_mse = mean_squared_error(y_test, y_test_pred)
    test_global_r2 = r2_score(y_test.reshape(-1), y_test_pred.reshape(-1))

    test_per_horizon_df = ML_function.evaluate_per_horizon(y_test, y_test_pred)

    print(f"TEST Global MSE: {test_global_mse:.6f}")
    print(f"TEST Global R2 : {test_global_r2:.6f}")

    pd.DataFrame({
        "test_global_mse": [test_global_mse],
        "test_global_r2": [test_global_r2]
    }).to_csv(f"test_results_XGB{Config}.csv", index=False)

    test_per_horizon_df.to_csv(f"test_per_horizon_XGB{Config}.csv", index=False)