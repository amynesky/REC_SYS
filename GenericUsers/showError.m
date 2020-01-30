close all;
clear all;

% path = 'training_prediction_actual_errors.txt';
% train = dlmread(path);
% 
% path = 'testing_prediction_actual_errors.txt';
% test=dlmread(path); 
% 
% train_sq = train.^2;
% test_sq = test.^2;
% 
% nnz_test = 0;
% nnz_train = 0;
% msq_test = 0;
% msq_train = 0;
% for it = 1 : length(test(:,3))
%     if test(it,3) ~= 0
%         nnz_test = nnz_test + 1;
%         msq_test = msq_test + (test_sq(it,3) - msq_test) / (nnz_test);
%        
%     end
%     if train(it,3) ~= 0
%         nnz_train = nnz_train + 1;
%         msq_train = msq_train + (train_sq(it,3) - msq_train) / (nnz_train);
%     end
% end
% msq_test
% msq_test = sum(test_sq);
% nnz_test = sum(test(:,3)~=0);
% msq_test = msq_test(3) / nnz_test
% 
% msq_train
% msq_train = sum(train_sq);
% nnz_train = sum(train(:,3)~=0);
% msq_train = msq_train(3) / nnz_train


path = '/Users/amynesky/pylon5/REC_SYS/GenericUsers/observations/ml-20m/not/'; 

path = strcat(path, '1_30_2020/meta_testing_error.txt');
training_error=dlmread(path);

its = [1:length(training_error)]';

p = polyfit(its, training_error, 1);
y = polyval(p,its);

figure
hold on
plot(its, training_error)
plot(its, y)

