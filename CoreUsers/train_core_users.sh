 #!/usr/bin/env sh

OBSERVATIONS=/pylon5/ac560rp/nesky/REC_SYS/CoreUsers

START=$(date +%s)
./core_users 2>&1 | tee $OBSERVATIONS/core_users_log.txt

END=$(date +%s)
DIFF=$(( $END - $START ))

echo "It took $DIFF seconds" #2>&1 | tee $OBSERVATIONS/core_users_train_time.txt


#while [ 1 ]; do echo -e '\a'; sleep 10; done


#while [ 1 ]; do say beep; sleep 30; done
#while [ 1 ]; do echo -en "\007"; sleep 10; done








