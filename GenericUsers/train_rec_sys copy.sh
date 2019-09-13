 #!/usr/bin/env sh

TOOLS=./build/tools

OBSERVATIONS=/pylon5/ac560rp/nesky/REC_SYS

START=$(date +%s)
./rec_sys 2>&1 | tee $OBSERVATIONS/rec_sys_log.txt

END=$(date +%s)
DIFF=$(( $END - $START ))

grep "TRAINING ERROR :" "$OBSERVATIONS/rec_sys_log.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_training_error_3.txt
awk 'BEGIN { FS="ERROR" }{ print $2 }' "$OBSERVATIONS/rec_sys_training_error_3.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_training_error_2.txt
awk '{ print $2 }' "$OBSERVATIONS/rec_sys_training_error_2.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_training_error.txt

grep "gpu_R_error total iterations :" "$OBSERVATIONS/rec_sys_log.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_error_iterations_3.txt
awk 'BEGIN { FS="iterations" }{ print $2 }' "$OBSERVATIONS/rec_sys_error_iterations_3.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_error_iterations_2.txt
awk '{ print $2 }' "$OBSERVATIONS/rec_sys_error_iterations_2.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_error_iterations_1.txt
awk 'NR%2==1' "$OBSERVATIONS/rec_sys_error_iterations_1.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_error_iterations.txt

grep "num_latent_factors =" "$OBSERVATIONS/rec_sys_log.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_latent_factors_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/rec_sys_latent_factors_3.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_latent_factors_2.txt
awk '{ print $1 }' "$OBSERVATIONS/rec_sys_latent_factors_2.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_latent_factors.txt

grep "gpu_R_error error :" "$OBSERVATIONS/rec_sys_log.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_testing_error_3.txt
awk 'BEGIN { FS=" error :" }{ print $2 }' "$OBSERVATIONS/rec_sys_testing_error_3.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_testing_error_2.txt
awk '{ print $1 }' "$OBSERVATIONS/rec_sys_testing_error_2.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_testing_error_1.txt
awk 'NR%2==0' "$OBSERVATIONS/rec_sys_testing_error_1.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_testing_error.txt


grep "full_ratingsMtx_dev_GA_current_batch_abs_max =" "$OBSERVATIONS/rec_sys_log.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_R_abs_max_3.txt
awk 'BEGIN { FS="=" }{ print $2 }' "$OBSERVATIONS/rec_sys_R_abs_max_3.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_R_abs_max_2.txt
awk '{ print $1 }' "$OBSERVATIONS/rec_sys_R_abs_max_2.txt" 2>&1 | tee $OBSERVATIONS/rec_sys_R_abs_max.txt

echo "It took $DIFF seconds" #2>&1 | tee $OBSERVATIONS/rec_sys_train_time.txt

rm $OBSERVATIONS/rec_sys_training_error_3.txt
rm $OBSERVATIONS/rec_sys_training_error_2.txt
rm $OBSERVATIONS/rec_sys_error_iterations_3.txt
rm $OBSERVATIONS/rec_sys_error_iterations_2.txt
rm $OBSERVATIONS/rec_sys_error_iterations_1.txt
rm $OBSERVATIONS/rec_sys_latent_factors_3.txt
rm $OBSERVATIONS/rec_sys_latent_factors_2.txt
rm $OBSERVATIONS/rec_sys_testing_error_3.txt
rm $OBSERVATIONS/rec_sys_testing_error_2.txt
rm $OBSERVATIONS/rec_sys_testing_error_1.txt
rm $OBSERVATIONS/rec_sys_R_abs_max_3.txt
rm $OBSERVATIONS/rec_sys_R_abs_max_2.txt





#while [ 1 ]; do echo -e '\a'; sleep 10; done


#while [ 1 ]; do say beep; sleep 30; done
#while [ 1 ]; do echo -en "\007"; sleep 10; done








