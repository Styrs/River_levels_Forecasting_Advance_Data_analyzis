
import pandas as pd
import data_clearing_function
import ML_function
import inspect, uuid, os



def run_LSTM_model(Config, path_to_csv,lookback_size=72,horizon_size=72):

    run_id = uuid.uuid4().hex[:8]

    stack = inspect.stack()
    caller = stack[1]
    caller_file = caller.filename
    caller_line = caller.lineno
    caller_function = caller.function

    print(f"\n==== NEW CALL TO run_LSTM_model ====")
    print(f"run_id={run_id}, pid={os.getpid()}")
    print(f"Called from: file='{caller_file}', line={caller_line}, function='{caller_function}'\n")
    Config = Config   # or "B", "C", "D" #Define what condition we use 
    path_to_csv = path_to_csv

    #############################################################################
    #Data import and final preparation
    #############################################################################


    data_station = ML_function.prepare_station_dataframe(path_to_csv, Config)

    #############################################################################
    #Run the LSTM model
    #############################################################################

    lookback_size = 72  # e.g., past 72 hours
    horizon_size = 72   # e.g., next 72 hours

    x_input, y_output = ML_function.make_windowed_data_NN(data_station, target_col="self_data", lookback=lookback_size, horizon=horizon_size)


    x_train, y_train, x_val, y_val, x_test, y_test = ML_function.split_time_series(x_input, y_output, train_frac=0.7, val_frac=0.2)

    print("Start training the LSTM model for one station")


    
    param_scope= {
    "units":                {"type": "int","bounds": (16, 96),},# Model capacity
    "num_layers":           {"type": "int","bounds": (1, 3),},
    "dropout":              {"type": "float","bounds": (0.0, 0.4),},# Regularisation
    "recurrent_dropout":    {"type": "float","bounds": (0.0, 0.3),},
    "batch_size":           {"type": "int","bounds": (32, 256),},# Training schedule
    "epochs":               {"type": "int","bounds": (3, 40),},
    "patience":             {"type": "int","bounds": (2, 8),},
    "learning_rate":        {"type": "float","bounds": (1e-4,3e-3),"prior": "log-uniform",},# Optimiser
}

    best_model_LSTM, best_params_LSTM, best_global_mse_LSTM, all_results_LSTM, results_df_LSTM, per_horizon_df_LSTM = ML_function.find_best_LSTM_model(x_train, y_train, x_val, y_val,
                                                                                                    param_scope,n_random_start=5,n_echo=30,random_state=0)


    print(best_model_LSTM)
    results_df_LSTM.to_csv("results_LSTM.csv", index=False)
    per_horizon_df_LSTM.to_csv("per_horizon_LSTM.csv", index=False)

    # ============================================
    # Evaluate BEST LSTM model on TEST SET
    # ============================================
    print("\nEvaluating best LSTM model on TEST SET...")

    y_test_pred = best_model_LSTM.predict(x_test)

    from sklearn.metrics import mean_squared_error, r2_score

    test_global_mse = mean_squared_error(y_test, y_test_pred)
    test_global_r2 = r2_score(y_test.reshape(-1), y_test_pred.reshape(-1))

    test_per_horizon_df = ML_function.evaluate_per_horizon(y_test, y_test_pred)

    print(f"TEST Global MSE: {test_global_mse:.6f}")
    print(f"TEST Global R2 : {test_global_r2:.6f}")

    pd.DataFrame({
        "test_global_mse": [test_global_mse],
        "test_global_r2": [test_global_r2]
    }).to_csv(f"test_results_LSTM{Config}.csv", index=False)

    test_per_horizon_df.to_csv(f"test_per_horizon_LSTM{Config}.csv", index=False)