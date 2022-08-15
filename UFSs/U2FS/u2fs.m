function featsSelected = u2fs(data,numClus,numFeats2select,options)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  U2FS - Utility metric for Unsupervised feature selection
%  
%  Code for unsupervised feature selection using the U2FS method described
%  in "Utility metric for Unsupervised feature selection", 
%  Amalia Villa, Abhijith Mundanad Narayanan, Sabine Van Huffel, Alexander
%  Bertrand, Carolina Varon
%
%  Needs functions: sigHighDim,utiSelect
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  featsSelected = u2fs(data,numClus,numFeats2select,options)
%  
%  INPUTS: 
%       data                - Data matrix (N x d). N is the number of 
%                             samples and d the number of features/dimensions.
%
%       numClus             - Number of clusters present in the data
%
%       numFeats2select     - Number of features to be selected. Can be
%                             single number or vector, if several 
%                             'numFeats2select' are to be evaluated.
%
%       options             - Struct optional inputs
%           - k             - Number of neighbors in the graph. 
%                             Default is k = 5
%           - simType       - Type of similarity matrix: 'Binary','RBF';
%                             corresponding to KNN and Binary weighting or 
%                             the RBF kernel respectively. 
%                             Default is 'RBF'
%           - sigma         - Type of sigma used: 'mean','highDim'. 'mean'
%                             corresponds to the average of the distance 
%                             matrix, while 'highDim' is the approximation 
%                             proposed in the U2FS paper.
%                             Default is 'highDim'
%
%  OUTPUTS: 
%       featsSelected       - Indices of features selected
%
%
%  Amalia Villa - amalia.villagomez@kuleuven.be
%  KU Leuven
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Verify inputs
% Neighbors building similarity graph
if ~isfield(options,'k')
    neighs = 5;
end

% Type of similarity matrix
if ~isfield(options,'simType')
    simType = lower('RBF');
else
    simType = lower(options.simType);
end

% Sigma value to use
if ~isfield(options,'sigma')
    sigma = 'highdim';
else
    sigma = lower(options.sigma);
end

%% Manifold learning

% 1 - Build similarity matrix
switch simType
    % KNN + binary weighting
    case {lower('Binary')}
        % Find nearest neighbours points
        locNN = knnsearch(data,data,'K',neighs + 1,'Distance','euclidean');
        % Build matrix of positions, to fill sparse matrix
        ind = repmat(locNN(:,1),neighs,1);
        ind(:,2) = reshape(locNN(:,2:neighs+1),[],1);
        % Build similarity matrix
        W = sparse(ind(:,1),ind(:,2),ones(size(ind,1),1),size(data,1),size(data,1));
        % Diagonal set to ones
        W = sparse(W{1,1} + eye(size(data,1),size(data,1)));
        % Guarantee symmetry
        W = max(W{1,1},W{1,1}'); 
    
    % RBF kernel
    case {lower('RBF')}
        
        % Select sigma
        switch sigma
            % Mean as used in MCFS
            case {lower('Mean')}
                
                % sig = mean of euclidean distance
                distMat = euDist(data);
                sig = mean(mean(distMat));
                sig = 2*sig^2;
                
                % Build similarity matrix
                dd = sum(data.^2,2);
                onev = ones(size(data,1),1);
                % Quick calculation affinity
                W = exp((2*(data*data') - dd*onev' - onev*dd')/sig);
                % Guarantee symmetry
                W = max(W,W');
            
            % Robust approximation for High dimensional data
            case {lower('highDim')}
                % Estimate sigma
                sig = sigHighDim(data);
                
                % Build similarity matrix
                dd = sum(data.^2,2);
                onev = ones(size(data,1),1);
                % Quick calculation affinity
                W = exp((2*(data*data') - dd*onev' - onev*dd')/sig);
                % Guarantee symmetry
                W = max(W,W');
        end
end


% 2 - Build Laplacian 

W(logical(eye(size(W)))) = 0;
% Calculate Degree matrix
D_mhalf = full(sum(W,2).^(-0.5));
% To avoid problems with Nans
posInf = (D_mhalf==inf);
D_mhalf(D_mhalf == Inf) = 0;
D_mhalf(posInf) = max(D_mhalf)*100;
% Diag
D = spdiags(D_mhalf,0,size(W,1),size(W,1));

% Normalized Laplacian
normLap = full(D)*W*full(D);
normLap = max(normLap,normLap');

% 3 - Extract eigenvectors
opts.disp=0;
opts.tol = 1e-3;
rng('default')
[eigenvects, ~] = eigs(normLap,numClus+1,'la',opts);
eigenvects = eigenvects(:,2:numClus+1);
    
% D ^ 1/2
D_mhalf = full(sum(W,2).^(0.5));
% To avoid problems with Nans
posInf = D_mhalf==inf;
D_mhalf(D_mhalf== Inf) = 0;
D_mhalf(posInf) = max(D_mhalf)*100;
% Diag
D = spdiags(D_mhalf,0,size(W,1),size(W,1));

% Alpha - Embedding E
alphas = full(D)*eigenvects;
    
%% Subset selection - Utility metric

featsSelected = utiSelect(data, alphas, numFeats2select);

