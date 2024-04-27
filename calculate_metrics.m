% This code is based on https://github.com/bilityniu/underimage-fusion-enhancement

clc;
clear all;
close all;



imgpath = './prediction';
imgdir = dir([imgpath,'*.png']);

count = 0;
uiqm_sum = 0;
uciqe_sum = 0;
for i = 1:length(imgdir)
    imgdir(i).name
    img = imread([imgpath imgdir(i).name]);
    
    uiqm = UIQM(img)
    uciqe = UCIQE(img)
    uiqm_sum = uiqm_sum + uiqm;
    uciqe_sum = uciqe_sum + uciqe;
    count = count + 1;

    mean_uiqm = uiqm_sum/count;
    mean_uciqe = uciqe_sum/count;
    
%     imshow(img);
end

disp("平均uiqm是");
disp(mean_uiqm);
disp("平均uciqe是");
disp(mean_uciqe);


