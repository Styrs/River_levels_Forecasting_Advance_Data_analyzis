import tree_model
import XGBoost
import GRU
import LSTM
import result_table_analyse
import pandas as pd


###############################################################################
# manualy control the number of heart use by the CPU
###############################################################################

#import tensorflow as tf

#tf.config.threading.set_intra_op_parallelism_threads(11)
#tf.config.threading.set_inter_op_parallelism_threads(11)

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense



config_list = ["A", "B"] #,"C","D"]

path_to_csv = "Data_debit/clean_data_individual/ARB_prepared.csv"

###############################################################################
# run the models for each config
###############################################################################


for Config in config_list:
    print(f"Running models for Config={Config}")

    print("Running the tree model")
    tree_model.run_tree_model(Config, path_to_csv, 72, 72)

    print("Tree model run completed")



    print("Running the XGBoost model")
    XGBoost.run_XGBoost_model(Config, path_to_csv, 72, 72)

    print("XGBoost model run completed")




    print("Running the GRU model")
    GRU.run_GRU_model(Config, path_to_csv, 72, 72)

    print("GRU model run completed")


    print("Running the LSTM model")
    LSTM.run_LSTM_model(Config, path_to_csv, 72, 72)

    print("LSTM model run completed")

###############################################################################
# Merge the results together for each config
###############################################################################
result_table_analyse.merge_results_together("A")
result_table_analyse.merge_results_together("B")

###############################################################################
# Horizon OLS: 
###############################################################################

res_A = result_table_analyse.analyze_config(
    "Results/Config_A/merged_test_per_horizon/merged_test_per_horizon_A.csv",
    config_name="A"
)

res_B = result_table_analyse.analyze_config(
    "Results/Config_B/merged_test_per_horizon/merged_test_per_horizon_B.csv",
    config_name="B"
)

final_results_OLS_perHorizon = pd.concat([res_A, res_B], ignore_index=True)
final_results_OLS_perHorizon.to_csv("Results/horizon_effect_summary.csv", index=False)

print(final_results_OLS_perHorizon)




###############################################################################
# Hyperparameter OLS: effect on (1) global_mse and (2) var_mse_across_horizons
###############################################################################

result_table_analyse.run_hyperparam_ols_for_config(
    config_name="A",
    path_to_csv="Results/Config_A/merged_results/merged_results_A.csv",
    out_root="Results",
    robust=True,
)

result_table_analyse.run_hyperparam_ols_for_config(
    config_name="B",
    path_to_csv="Results/Config_B/merged_results/merged_results_B.csv",
    out_root="Results",
    robust=True,
)

###############################################################################
# Plot the results
###############################################################################


# results of MSE per horizon
result_table_analyse.plot_best_models_MSE_per_horizon(Config= "A")
result_table_analyse.plot_best_models_MSE_per_horizon(Config= "B")

result_table_analyse.plot_best_models_R2_per_horizon(Config= "A")
result_table_analyse.plot_best_models_R2_per_horizon(Config= "B")

result_table_analyse.plot_best_models_historgram(variable= "global_mse")
result_table_analyse.plot_best_models_historgram(variable= "global_r2")