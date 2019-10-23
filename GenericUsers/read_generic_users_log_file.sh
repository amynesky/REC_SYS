 #!/usr/bin/env sh

OBSERVATIONS=/pylon5/ac560rp/nesky/REC_SYS/GenericUsers

START=$(date +%s)


# grep "TRAINING ERROR :" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error_3.txt
# awk 'BEGIN { FS="ERROR" }{ print $2 }' "$OBSERVATIONS/generic_users_training_error_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error_2.txt
# awk '{ print $2 }' "$OBSERVATIONS/generic_users_training_error_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error.txt


grep "gpu_R_error_training error :" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error_3.txt
awk 'BEGIN { FS=" error :" }{ print $2 }' "$OBSERVATIONS/generic_users_training_error_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_training_error_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error.txt

grep "gpu_R_error_testing error on training entries:" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_on_training_entries_3.txt
awk 'BEGIN { FS=" entries:" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_error_on_training_entries_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_on_training_entries_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_testing_error_on_training_entries_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_on_training_entries.txt

grep "gpu_R_error_testing error on testing entries:" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_on_testing_entries_3.txt
awk 'BEGIN { FS=" entries:" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_error_on_testing_entries_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_on_testing_entries_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_testing_error_on_testing_entries_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_on_testing_entries.txt

grep "gpu_R_error_training total iterations :" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error_iterations_3.txt
awk 'BEGIN { FS="iterations" }{ print $2 }' "$OBSERVATIONS/generic_users_training_error_iterations_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error_iterations_2.txt
awk '{ print $2 }' "$OBSERVATIONS/generic_users_training_error_iterations_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_training_error_iterations.txt

grep "gpu_R_error_testing total iterations :" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_iterations_3.txt
awk 'BEGIN { FS="iterations" }{ print $2 }' "$OBSERVATIONS/generic_users_testing_error_iterations_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_iterations_2.txt
awk '{ print $2 }' "$OBSERVATIONS/generic_users_testing_error_iterations_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_testing_error_iterations.txt

grep "num_latent_factors =" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_latent_factors_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/generic_users_latent_factors_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_latent_factors_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_latent_factors_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_latent_factors.txt

grep "full_ratingsMtx_dev_GU_current_batch_abs_max =" "$OBSERVATIONS/generic_users_log.txt" 2>&1 | tee $OBSERVATIONS/generic_users_R_abs_max_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/generic_users_R_abs_max_3.txt" 2>&1 | tee $OBSERVATIONS/generic_users_R_abs_max_2.txt
awk '{ print $1 }' "$OBSERVATIONS/generic_users_R_abs_max_2.txt" 2>&1 | tee $OBSERVATIONS/generic_users_R_abs_max.txt


rm $OBSERVATIONS/generic_users_training_error_3.txt
rm $OBSERVATIONS/generic_users_training_error_2.txt
rm $OBSERVATIONS/generic_users_training_error_iterations_3.txt
rm $OBSERVATIONS/generic_users_training_error_iterations_2.txt
rm $OBSERVATIONS/generic_users_testing_error_iterations_3.txt
rm $OBSERVATIONS/generic_users_testing_error_iterations_2.txt
rm $OBSERVATIONS/generic_users_testing_error_on_testing_entries_3.txt
rm $OBSERVATIONS/generic_users_testing_error_on_testing_entries_2.txt
rm $OBSERVATIONS/generic_users_testing_error_on_training_entries_3.txt
rm $OBSERVATIONS/generic_users_testing_error_on_training_entries_2.txt
rm $OBSERVATIONS/generic_users_latent_factors_3.txt
rm $OBSERVATIONS/generic_users_latent_factors_2.txt
rm $OBSERVATIONS/generic_users_testing_error_3.txt
rm $OBSERVATIONS/generic_users_testing_error_2.txt
rm $OBSERVATIONS/generic_users_R_abs_max_3.txt
rm $OBSERVATIONS/generic_users_R_abs_max_2.txt


END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds" #2>&1 | tee $OBSERVATIONS/generic_users_train_time.txt




echo -e '\a'

#while [ 1 ]; do echo -e '\a'; sleep 10; done


#while [ 1 ]; do say beep; sleep 30; done
#while [ 1 ]; do echo -en "\007"; sleep 10; done








