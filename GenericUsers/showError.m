close all;
clear all;

path = 'full_ratingsMtx_GU.txt';
R_ = dlmread(path);

path = 'V.txt';
V=dlmread(path); 

path = 'U_GU.txt';
U=dlmread(path); 



R = U*V';