function ranking = LLCFS(X,nClusters)
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
%  ------------------------------------------------------------------------
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
% Please refer to
%  Zeng, H. and Cheung, Y.M., 2010. Feature selection and kernel learning 
%       for local learning-based clustering. IEEE Transactions on Pattern 
%           Analysis and Machine Intelligence, 33(8), pp.1532-1547.
%  ------------------------------------------------------------------------
% Input
%	X: nSmp * nDim
%	param, a struct of parameters
%		nClusters, the number of clusters
%		k, the size of knn
%		beta, the regularization parameter
% Output
%	Y: nSmp * nClusters
%	tao: nDim * 1

if nargin <2  
    nClusters =2;
end
%===================setup=======================
knnCandi = 5;
graphTypeCandi = 1;
betaCandidates = 10.^-1;
paramcell = llcfs_build_param(nClusters, knnCandi, betaCandidates, graphTypeCandi );
%===============================================
param = paramcell{1};
X = double(X);

if isfield(param, 'nClusters')
    c = param.nClusters;
end

k = 30;
if isfield(param, 'k')
    k = param.k;
end

beta = 1;
if isfield(param, 'beta')
    beta = param.beta;
end

kType = 1;
if isfield(param, 'kType')
    kType = param.kType;
end

maxiter = 2;
if isfield(param, 'maxiter')
    maxiter = param.maxiter;
end

epsilon = 1e-5;
if isfield(param, 'epsilon')
    epsilon = param.epsilon;
end

isTao = 0;
epsilon_tao = 1e-5;
[n, d] = size(X);


% convergence by maxiter
isMaxiter = 1;
if maxiter > 0
    isMaxiter = 1;
end

% convergence by epsilon
isEpsilon = 0;
if isEpsilon > 0
    isEpsilon = 1;
end

tao = ones(d,1) / d;

objHistory = [];
iter = 0;
while true
    
    wX = bsxfun(@times, X, sqrt(max(tao, eps))' );
    wX2 = bsxfun(@times, X, max(tao, eps)' );
    wK = wX * wX';
    % k-mutual neighbors re-computation using weighted features
    switch kType
        case 1
            W = SimGraph_NearestNeighbors(wX', k, 2, 0);
            [idx, jdx, ~] = find(W);
            kIdx = cell(n, 1);
            nz = length(idx);
            for ii = 1:nz
            	kIdx{jdx(ii)} = [kIdx{jdx(ii)}, idx(ii)];
            end
        case 2
            if isempty(which('knnsearch'))
                disp('The funcion knnsearch in stat toolbox is not found');
            else
                [kIdx, ~] = knnsearch(wX, wX, 'k', min(n, k + 1) );
                kIdx = kIdx(:, 2:end);
                kIdx = mat2cell(kIdx, ones(n, 1), size(kIdx, 2));
            end
        otherwise
            disp('');
    end
    
    % construct A for laplacian
    A = zeros(n);
    wA = cell(n,1);% pre storage for w computation
    for i = 1:n
        lidx = kIdx{i};
        ni = length(lidx);
        if ni > 1
            Ki = wK(lidx, lidx);
            ki = wK(i, lidx);
            Hi = eye(ni) - ones(ni, ni) / ni;
            Ii = eye(ni);
            Iib = Ii / beta;
            Ai = Hi * Ki * Hi;
            Ai = (Ai + Iib) \ Ai;
            Ai = Hi - Hi * Ai;
            Ai = Ai * beta;
            wA{i} = wX2(lidx, :)' * Ai; % EQ 15
            Ai = (ki - sum(Ki) / ni) * Ai;
            Ai = Ai + ones(1, ni) / ni;
            A(i, lidx) = Ai;
        end
    end
    
    % construct laplacian for local learning
    M = eye(n) - A;
    M = M' * M;
    M(isnan(M)) = 0;
	M(isinf(M)) = 0;
	
    % first c eigenvectors corresponding to the first c smallest eigenvalues
    M = (M + M') / 2;
    [Y, eigval] = eig(M);
    eigval = diag(eigval);
    [eigval, eigidx] = sort(eigval, 'ascend');
	eigval = eigval(eigidx(1:c));
    Y = Y(:, eigidx(1:c));
    
    objHistory = [objHistory; sum(eigval)];%#ok
    
	
    % compute wc to compute tao
    tao_old = tao;
	
	tao = zeros(d, 1);
    for i = 1:n
        lidx = kIdx{i};
        ni = length(lidx);
        if ni > 1
            wi = wA{i} * Y(lidx,:);
            tao = sum(wi.^2, 2) + tao;
        end
    end
	tao = sqrt(tao);
    tao = tao / sum(tao);
    
    % check the convergence
    iter = iter + 1;
    if isEpsilon && iter > 1
        if abs(objHistory(end-1) - objHistory(end)) < epsilon
            break;
        end
    end
	if isTao && sum(abs(tao_old - tao)) < epsilon_tao
		break;
	end
    if isMaxiter && iter == maxiter
        break;
    end
end

[~, ranking] = sort(tao, 'ascend');
ranking = ranking';
end

