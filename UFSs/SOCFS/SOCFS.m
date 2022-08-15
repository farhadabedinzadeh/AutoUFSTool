function [ranking] = SOCFS(X, m, lambda1, lambda2, ITER1, ITER2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for the Auto-UFSTool, which is an Automatic Unspervised %
% Feature Selection Toolbox                                                %
% Version 1.0 August 2022   |  Copyright (c) 2022   | All rights reserved  %
%        ------------------------------------------------------            %
% Redistribution and use in source and binary forms, with or without       %
% modification, are permitted provided that the following conditions are   %
% met:                                                                     %
%    * Redistributions of source code must retain the above copyright      %
%    * Redistributions in binary form must reproduce the above copyright.  %
%         ------------------------------------------------------           %
% "Auto-UFSTool,                                                           %
%  An Automatic MATLAB Toolbox for Unsupervised Feature Selection"         %
%         ------------------------------------------------------           %
%                    Farhad Abedinzadeh | Yegane Modaresnia                %
%          farhaad.abedinzade@gmail.com | y.modaresnia@yahoo.com           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% This code is for the algorithm described in 
% "Unsupervised Simultaneous Orthogonal Basis Clustering Feature Selection" (CVPR 2015)
% Revised Aug, 2015 by Dongyoon Han 

%% Problem:
% min_W,b,B,E ||X'W + b1' - EB'||^2 + lambda1 * ||W||_21
% s.t. B'B = I, E'E = I, E>=0
%% Input and output:
% [Input]
% X: the data matrix
% E: the encoding matix 
% B: the orthogonal basis matrix
% W, b: the feature selection matrix and the offset vector
%
% [Output]
% feature_idx: indices of selected features
% --------------------------------------------------

 
X = X'; % NOTE
if nargin < 2
    m = size(X,1);
end

if nargin < 3
    lambda1 = 100;
    m = size(X,1);
end

if nargin < 4
    lambda2 = 100;
    m = size(X,1);
end

if nargin < 5
    ITER1 = 50;
    m = size(X,1);
end

if nargin < 6
    ITER2 = 50;
    m = size(X,1);
end

% -----------------------------------------------    
[d, n] = size(X);
d2 = ones(d, 1);
X = X - mean(X,2)* ones(1,n); 
E = orth(rand(n,m));
B = orth(rand(m,m));
W = rand(d,m);

H = rand(size(E));
b = zeros(m, 1);
XX = X * X';

for iter = 1 : ITER1    
    iter2 = 1;
    WXB = (W' * X + b * ones(1, n))' * B;
    while(iter2 <= ITER2)               
        [LE, ~, RE] = svd(WXB + lambda2 * H, 'econ'); E = LE * RE';
        H = 0.5 * (E + abs(E));   
        iter2 = iter2 + 1;    
    end
     
    AB = E' * (W' * X + b * ones(1, n))';
    [LE, ~, RE] = svd(AB', 'econ');
    B = LE * RE'; 
    
    EB = E * B';
    D2 = spdiags(d2, 0, d, d);
    W = (XX + lambda1 * D2) \ (X * (EB - ones(n, 1) * b'));
%      b = mean(EB' - W'*X, 2);
    d2 = 1./ (2 * (sqrt(sum(W .* W, 2) + eps)));
end

[~, idx] = sort(sum(W.*W,2), 'ascend');
ranking = idx';
end