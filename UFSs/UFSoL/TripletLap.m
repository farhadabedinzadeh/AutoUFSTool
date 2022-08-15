function [C, L] = TripletLap(data,k) %(scale non-zero weights to [0,1])
%% Constructing an unsupervised graph based on triplets
% Input:
%       data    -each column is a data point
%       k       -number of nearest neighbors
% Output:
%       C       -weighting matrix
%       L       -Laplacian matrix


%% Initialization
[~, nSmp] = size(data);
if k <= 0  || k >= nSmp
    error('Parameter k is error!');
end    

%% distance and weighting
Dist = EuDist2(data',[],1); % Euclidean distance
[~, idx] = sort(Dist,2); % sort each row 'ascend'
idx = idx(:,2:k+1); % default: not self-connected
G = sparse(repmat([1:nSmp]',[k,1]),idx(:),ones(numel(idx),1),nSmp,nSmp);
G = full(G);
% i^th row of G: Among the other (nSmp-1) samples, which belongs to 
% the i^th sample's k nearest neighbors  =1: belong; =0: not belong
C = Dist.*G; 
C = -mean(sum(G,2))*bsxfun(@minus,C,sum(C,2)./sum(G,2)); 
C = C.*G;      % sum(G,2): [k,k,...,k]'
%%%%% the following two lines aim to scale non-zero weights to [0,1] %%%%
C = bsxfun(@rdivide,bsxfun(@minus,C,min(C,[],2)),max(C,[],2)-min(C,[],2));
C = C.*G;      % the scaling is row-wise
%%%%% For the above two lines, users can comment out by real demand %%%%%
clear Dist G idx % clear useless variable
C = full(C);

%% obtain Laplacian matrix
Csym = full(0.5*(C+C')); 
d = sum(Csym,2);
if min(diag(d))>0    
    % Normalized Laplacian Matrix
    L = diag(1./sqrt(d))*(diag(d)-Csym)*diag(1./sqrt(d)); 
else
    % Unnormalized Laplacian Matrix
    L = diag(d)-Csym;     
end
L = full(max(L,L')); % guarantee symmetry