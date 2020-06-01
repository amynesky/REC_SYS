 #!/usr/bin/env sh

OBSERVATIONS=/pylon5/ac560rp/nesky/REC_SYS/CoreUsers/observations/ml-20m/dont/user_cosine_similarity_incorporates_ratings/frequency_based
LOGFILE=slurm-9111114.out

START=$(date +%s)


grep "gpu_R_error MSQER on training entries:" "$OBSERVATIONS/$LOGFILE" 2>&1 | tee $OBSERVATIONS/core_users_training_error_3.txt
awk 'BEGIN { FS="entries:" }{ print $2 }' "$OBSERVATIONS/core_users_training_error_3.txt" 2>&1 | tee $OBSERVATIONS/core_users_training_error_2.txt
awk '{ print $2 }' "$OBSERVATIONS/core_users_training_error_2.txt" 2>&1 | tee $OBSERVATIONS/core_users_training_error.txt

# grep "gpu_R_error_testing error :" "$OBSERVATIONS/$LOGFILE" 2>&1 | tee $OBSERVATIONS/core_users_testing_error_3.txt
# awk 'BEGIN { FS=" error :" }{ print $2 }' "$OBSERVATIONS/core_users_testing_error_3.txt" 2>&1 | tee $OBSERVATIONS/core_users_testing_error_2.txt
# awk '{ print $1 }' "$OBSERVATIONS/core_users_testing_error_2.txt" 2>&1 | tee $OBSERVATIONS/core_users_testing_error.txt

# grep "gpu_R_error_testing total iterations :" "$OBSERVATIONS/$LOGFILE" 2>&1 | tee $OBSERVATIONS/core_users_testing_error_iterations_3.txt
# awk 'BEGIN { FS="iterations" }{ print $2 }' "$OBSERVATIONS/core_users_testing_error_iterations_3.txt" 2>&1 | tee $OBSERVATIONS/core_users_testing_error_iterations_2.txt
# awk '{ print $2 }' "$OBSERVATIONS/core_users_testing_error_iterations_2.txt" 2>&1 | tee $OBSERVATIONS/core_users_testing_error_iterations.txt

# grep "num_latent_factors =" "$OBSERVATIONS/$LOGFILE" 2>&1 | tee $OBSERVATIONS/core_users_latent_factors_3.txt
# awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/core_users_latent_factors_3.txt" 2>&1 | tee $OBSERVATIONS/core_users_latent_factors_2.txt
# awk '{ print $1 }' "$OBSERVATIONS/core_users_latent_factors_2.txt" 2>&1 | tee $OBSERVATIONS/core_users_latent_factors.txt

# grep "full_ratingsMtx_dev_CU_current_batch_abs_max =" "$OBSERVATIONS/$LOGFILE" 2>&1 | tee $OBSERVATIONS/core_users_R_abs_max_3.txt
# awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/core_users_R_abs_max_3.txt" 2>&1 | tee $OBSERVATIONS/core_users_R_abs_max_2.txt
# awk '{ print $1 }' "$OBSERVATIONS/core_users_R_abs_max_2.txt" 2>&1 | tee $OBSERVATIONS/core_users_R_abs_max.txt

rm $OBSERVATIONS/core_users_training_error_3.txt
rm $OBSERVATIONS/core_users_training_error_2.txt
# rm $OBSERVATIONS/core_users_testing_error_iterations_3.txt
# rm $OBSERVATIONS/core_users_testing_error_iterations_2.txt
# rm $OBSERVATIONS/core_users_latent_factors_3.txt
# rm $OBSERVATIONS/core_users_latent_factors_2.txt
# rm $OBSERVATIONS/core_users_testing_error_3.txt
# rm $OBSERVATIONS/core_users_testing_error_2.txt
# rm $OBSERVATIONS/core_users_R_abs_max_3.txt
# rm $OBSERVATIONS/core_users_R_abs_max_2.txt


END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" #2>&1 | tee $OBSERVATIONS/core_users_train_time.txt






#while [ 1 ]; do echo -e '\a'; sleep 10; done


#while [ 1 ]; do say beep; sleep 30; done
#while [ 1 ]; do echo -en "\007"; sleep 10; done








