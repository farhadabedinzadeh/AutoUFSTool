function [W, L] = getS(data,k,num) 
% construct a graph in an unsupervised manner
% 
% Input:
%       data    -each column is a sample
%       k       -number of neighbors
%       num     -'0-1' weight when num==1; (default)
%                Gaussian weight, otherwise.
% Output:
%       W       -weighting matrix
%       L       -Laplacian matrix

%% Settings
[~, nSmp] = size(data);
if ~exist('k', 'var')
    k = nSmp;
end
if k < 1  || k > nSmp
    error('Parameter k is error!');
end  
if ~exist('num', 'var')
    num = 1; % 0-1 weight
end

%% Run
Dist = EuDist2(data',[],0); % Euclidean distance.^2
sigma = 4*mean(mean(Dist)); % =1e2 (optional);
[~, idx] = sort(Dist,2); % sort each row ascend
idx = idx(:,1:k); % default: self-connected
G = sparse(repmat([1:nSmp]',[k,1]),idx(:),ones(numel(idx),1),nSmp,nSmp);
%%% the i_th row of matrix G stores the information of the  
%%% i_th sample's k neighbors. (1: is a neighbor, 0: is not)
if num ~= 1
    W = (exp(-Dist/sigma)).*G; % Gaussian kernel weight
    W = full(0.5*(W+W')); % guarantee symmetry
else
    W = G; % 0-1 weight
    W = full(max(W,W')); % guarantee symmetry
end
clear idx Dist G  % useless

%% get L
d = sum(W,2);
if min(diag(d))>0
    L = diag(1./sqrt(d))*(diag(d)-W)*diag(1./sqrt(d)); 
    % this is: Normalized Laplacian Matrix
else
    L = diag(d)-W; 
    % this is: Unnormalized Laplacian Matrix
end
clear d % useless
L = full(max(L,L')); % guarantee symmetry