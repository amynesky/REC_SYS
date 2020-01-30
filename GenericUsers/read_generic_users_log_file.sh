 #!/usr/bin/env sh

OBSERVATIONS=/pylon5/ac560rp/nesky/REC_SYS/GenericUsers
#OBSERVATIONS=$OBSERVATIONS/observations/ml-20m/not/1_percent_GU/trial_2

FILENAME=generic_users_log.txt
#FILENAME=slurm-6611016.out

START=$(date +%s)


# grep "TRAINING ERROR :" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_training_MSQER_3.txt
# awk 'BEGIN { FS="ERROR" }{ print $2 }' "$OBSERVATIONS/generic_users_training_MSQER_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_MSQER_2.txt
# awk '{ print $2 }' "$OBSERVATIONS/generic_users_training_MSQER_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_MSQER.txt


grep "gpu_R_error_training error :" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_training_MSQER_3.txt
awk 'BEGIN { FS=" error :" }{ print $2 }' "$OBSERVATIONS/generic_users_training_MSQER_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_MSQER_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_training_MSQER_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_MSQER.txt

grep "gpu_R_error_testing MSQER on training entries:" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_testing_MSQER_on_training_entries_3.txt
awk 'BEGIN { FS=" entries:" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_MSQER_on_training_entries_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_MSQER_on_training_entries_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_testing_MSQER_on_training_entries_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_MSQER_on_training_entries.txt

grep "gpu_R_error_testing MSQER on testing entries:" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_testing_MSQER_on_testing_entries_3.txt
awk 'BEGIN { FS=" entries:" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_MSQER_on_testing_entries_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_MSQER_on_testing_entries_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_testing_MSQER_on_testing_entries_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_MSQER_on_testing_entries.txt

grep "Testing error norm over E" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_testing_normalized_over_rand_error_3.txt
awk 'BEGIN { FS="|]:" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_normalized_over_rand_error_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_normalized_over_rand_error_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_testing_normalized_over_rand_error_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_normalized_over_rand_error.txt

grep "Testing error norm over norm of testing only entries:" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_testing_normalized_error_3.txt
awk 'BEGIN { FS=" entries:" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_normalized_error_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_normalized_error_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_testing_normalized_error_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_normalized_error.txt

grep "gpu_R_error_training total iterations :" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_training_iterations_3.txt
awk 'BEGIN { FS="iterations" }{ print $2 }' "$OBSERVATIONS/generic_users_training_iterations_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_iterations_2.txt
awk '{ print $2 }' "$OBSERVATIONS/generic_users_training_iterations_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_iterations.txt

grep "gpu_R_error_testing total iterations :" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_testing_iterations_3.txt
awk 'BEGIN { FS="iterations" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_iterations_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_iterations_2.txt
awk '{ print $2 }' "$OBSERVATIONS/generic_users_testing_iterations_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_iterations.txt

grep "num_latent_factors =" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_latent_factors_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/generic_users_latent_factors_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_latent_factors_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_latent_factors_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_latent_factors.txt

grep ": R_GU maximum absolute value =" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_R_abs_max_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/generic_users_R_abs_max_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_R_abs_max_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_R_abs_max_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_R_abs_max.txt

grep "delta R_GU maximum absolute value =" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_delta_R_abs_max_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/generic_users_delta_R_abs_max_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_delta_R_abs_max_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_delta_R_abs_max_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_delta_R_abs_max.txt

grep "delta U maximum absolute value =" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_delta_U_abs_max_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/generic_users_delta_U_abs_max_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_delta_U_abs_max_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_delta_U_abs_max_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_delta_U_abs_max.txt

grep "delta V maximum absolute value =" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_delta_V_abs_max_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/generic_users_delta_V_abs_max_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_delta_V_abs_max_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_delta_V_abs_max_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_delta_V_abs_max.txt

grep "err norm when clustering over training norm:" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_k_mean_er_normalized_3.txt
awk 'BEGIN { FS=" norm:" }{ print $2 }' "$OBSERVATIONS/generic_users_k_mean_er_normalized_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_k_mean_er_normalized_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_k_mean_er_normalized_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_k_mean_er_normalized.txt

grep "mean sqed err when clustering :" "$OBSERVATIONS/$FILENAME" 2>&1 | tee $OBSERVATIONS/generic_users_k_mean_MSQER_3.txt
awk 'BEGIN { FS="clustering :" }{ print $2 }' "$OBSERVATIONS/generic_users_k_mean_MSQER_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_k_mean_MSQER_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_k_mean_MSQER_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_k_mean_MSQER.txt


rm $OBSERVATIONS/generic_users_training_MSQER_3.txt
rm $OBSERVATIONS/generic_users_training_MSQER_2.txt
rm $OBSERVATIONS/generic_users_training_iterations_3.txt
rm $OBSERVATIONS/generic_users_training_iterations_2.txt

rm $OBSERVATIONS/generic_users_testing_iterations_3.txt
rm $OBSERVATIONS/generic_users_testing_iterations_2.txt
rm $OBSERVATIONS/generic_users_testing_MSQER_on_testing_entries_3.txt
rm $OBSERVATIONS/generic_users_testing_MSQER_on_testing_entries_2.txt
rm $OBSERVATIONS/generic_users_testing_MSQER_on_training_entries_3.txt
rm $OBSERVATIONS/generic_users_testing_MSQER_on_training_entries_2.txt
rm $OBSERVATIONS/generic_users_testing_normalized_over_rand_error_3.txt
rm $OBSERVATIONS/generic_users_testing_normalized_over_rand_error_2.txt
rm $OBSERVATIONS/generic_users_testing_normalized_error_3.txt
rm $OBSERVATIONS/generic_users_testing_normalized_error_2.txt

rm $OBSERVATIONS/generic_users_latent_factors_3.txt
rm $OBSERVATIONS/generic_users_latent_factors_2.txt

rm $OBSERVATIONS/generic_users_k_mean_MSQER_3.txt
rm $OBSERVATIONS/generic_users_k_mean_MSQER_2.txt
rm $OBSERVATIONS/generic_users_k_mean_er_normalized_3.txt
rm $OBSERVATIONS/generic_users_k_mean_er_normalized_2.txt
rm $OBSERVATIONS/generic_users_R_abs_max_3.txt
rm $OBSERVATIONS/generic_users_R_abs_max_2.txt
rm $OBSERVATIONS/generic_users_delta_R_abs_max_3.txt
rm $OBSERVATIONS/generic_users_delta_R_abs_max_2.txt
rm $OBSERVATIONS/generic_users_delta_U_abs_max_3.txt
rm $OBSERVATIONS/generic_users_delta_U_abs_max_2.txt
rm $OBSERVATIONS/generic_users_delta_V_abs_max_3.txt
rm $OBSERVATIONS/generic_users_delta_V_abs_max_2.txt


END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" #2>&1 | tee $OBSERVATIONS/generic_users_train_time.txt




#echo -e '\a'

#while [ 1 ]; do echo -e '\a'; sleep 10; done


#while [ 1 ]; do say beep; sleep 30; done
#while [ 1 ]; do echo -en "\007"; sleep 10; done








