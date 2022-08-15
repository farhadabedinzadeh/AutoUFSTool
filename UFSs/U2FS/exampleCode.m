%% Example code

close all;
clearvars;
clc;

% Load data
load cloudsExample;

%% Inputs
% Data
data = simul.clouds.fea;
% 3 clouds
numClus = 3;
% 2 relevant original features to be selected
numFeats2select = 2; 
% Options - kernel embedding, with proposed sigma
options.simType = 'Binary';
options.sigma = 'Mean';

% Feature selection
featsSelected = u2fs(data,numClus,numFeats2select,options);

% Plot final features
figure()
scatter(data(:,featsSelected(1,1)),data(:,featsSelected(2,1)))