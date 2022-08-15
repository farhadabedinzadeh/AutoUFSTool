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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for the SRCFS method, which is proposed in the   %
% following paper:                                                  %
%                                                                   %
% Dong Huang, Xiaosha Cai, and Chang-Dong Wang.                     %
% "Unsupervised Feature Selection with Multi-Subspace Randomization %
% and Collaboration",                                               %
% Knowledge-Based Systems, in press, 2019.                          %
%                                                                   %
% The code has been tested in Matlab R2016b and Matlab R2018a.      %
% Written by Huang Dong. (huangdonghere@gmail.com)                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [rankings,feaWeights] = SRCFS(fea, para_K, para_s, para_m)

%% Input:
% fea:      the n*d data matrix with each row being a data sample.
% para_K:   the number of nearest neighobrs.
% para_s:   the number of random subspaces in each basic feature partition.
% para_m:   the number of basic feature partitions.
%% Output:
% feaWeights:	the feature weights computed by SRCFS.
% rankings:     the ranking of all features according to their weights.
%% Example:
% If you need to select D features from the dataset with a data matrix fea,
% you can perform SRCFS like this:
%       [rankings,~] = SRCFS(fea);
%       selectedFeatureIdx = rankings(1:D);
% Here, selectedFeatureIdx indicates the indices of D selected features.

if nargin < 4
    para_m = 10;
end
if nargin < 3
    para_s = 20;
end
if nargin < 2
    para_K = 5;
end

[~,d] = size(fea);
if para_s > 1
    [feas,feaIdxs] = genRandomFeaSets(fea,para_s,para_m);
    fWs = zeros(d,length(feas));
    for iFS = 1:length(feas)
        knnW = buildSimGraphKNN(feas{iFS},para_K);
        tmpWs = LaplacianScore(feas{iFS}, knnW);
        fWs(feaIdxs{iFS},iFS) = tmpWs;
    end
    feaWeights = sum(fWs,2)./para_m;
else
    knnW = buildSimGraphKNN(fea,para_K);
    [feaWeights] = LaplacianScore(fea, knnW);
end
[~,rankings] = sort(feaWeights,'descend');

function W = buildSimGraphKNN(data,K)
D = EuDist2(data,data,1);
[n,d] = size(data);

for i = 1:n
    D(i,i) = 1e100;
end

dump = zeros(n,K);
idx = dump;
for i = 1:K
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*n+[1:n]';
    D(temp) = 1e100; 
end

sigma = mean(mean(dump));
clear temp D

dump = exp(-(dump.^2)/(2*sigma^2));
Gsdx = dump; clear dump
Gidx = repmat([1:n]',1,K);
Gjdx = idx;
W=sparse(Gidx(:),Gjdx(:),Gsdx(:),n,n); clear Gidx Gjdx  Gsdx
W = max(W,W');

function [datas,dataIdxs] = genRandomFeaSets(data,para_s,para_m)
[n,d] = size(data);
d_each = max(ceil(d/para_s),1);
dataIdxs = [];
datas = [];
nextFSidx = 1;
for iM = 1:para_m
    rand('state',sum(100*clock)*rand(1));
    rDidxs = randperm(d);
    for iF = 1:para_s
        tmpIdx = rDidxs(((iF-1)*d_each+1):min((iF*d_each),end));
        if length(tmpIdx)==0
            break;
        end
        dataIdxs{nextFSidx} = tmpIdx;
        datas{nextFSidx} = data(:,tmpIdx);
        nextFSidx = nextFSidx+1;
    end
end

function D = EuDist2(fea_a,fea_b,bSqrt)

if ~exist('bSqrt','var')
    bSqrt = 1;
end

if (~exist('fea_b','var')) || isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@plus,aa,aa') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end

    D = bsxfun(@plus,aa,bb') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end

function [Y] = LaplacianScore(X, W)
%	Usage:
%	[Y] = LaplacianScore(X, W)
%
%	X: Rows of vectors of data points
%	W: The affinity matrix.
%	Y: Vector of (1-LaplacianScore) for each feature.
%      The features with larger y are more important.
%
%    Examples:
%
%       fea = rand(50,70);
%       options = [];
%       options.Metric = 'Cosine';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'Cosine';
%       W = constructW(fea,options);
%
%       LaplacianScore = LaplacianScore(fea,W);
%       [junk, index] = sort(-LaplacianScore);
%       
%       newfea = fea(:,index);
%       %the features in newfea will be sorted based on their importance.
%
%	Type "LaplacianScore" for a self-demo.
%
% See also constructW
%
%Reference:
%
%   Xiaofei He, Deng Cai and Partha Niyogi, "Laplacian Score for Feature Selection".
%   Advances in Neural Information Processing Systems 18 (NIPS 2005),
%   Vancouver, Canada, 2005.   
%
%   Deng Cai, 2004/08

if nargin == 0, selfdemo; return; end

[nSmp,nFea] = size(X);

if size(W,1) ~= nSmp
    error('W is error');
end

D = full(sum(W,2));
L = W;

allone = ones(nSmp,1);


tmp1 = D'*X;

D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);

DPrime = sum((X'*D)'.*X)-tmp1.*tmp1/sum(diag(D));
LPrime = sum((X'*L)'.*X)-tmp1.*tmp1/sum(diag(D));

DPrime(find(DPrime < 1e-12)) = 10000;

Y = LPrime./DPrime;
Y = Y';
Y = full(Y);
