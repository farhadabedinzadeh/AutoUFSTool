function [rankx, W,V, valVec] = rank_fir_ordinal_locality(X,nClass,para) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Descripion: 
% This function implements Algorithm 1 of our paper
% Jun Guo, Yanqing Guo, Xiangwei Kong, and Ran He, 
% "Unsupervised Feature Selection with Ordinal Locality," In ICME 2017.
%   -Source code version 1.0  2017/04/25 by Jun Guo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: 
% Users need to set the struct 'para' in advance, e.g., 
%   para.p0 = 'sample';
%   para.p1 = 1e6;
%   para.p2 = 1e2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%       X        -each column is a sample; each row is a feature
%       nClass   -total number of clusters
%       para.p0  -'sample' & 'feature' -level ordinal locality
%                 'sample'-level is just the main content of our paper
%                 'feature'-level is proposed at the end of conclusion
%       para.p1  -regularization parameter: alpha*Tr(W'*...*W) 
%       para.p2  -regularization parameter: beta*||W||_2,1
% Output:
%       W        -projection matrix
%       V        -each column is a non-negative orthogonal coefficient
%       feaIdx   -indices of selected dimensions in original data
%       valVec   -storing objective function values in each iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Please cite our work if you find the code helpful.
% @inproceedings{JunGuo_ICME_2017,
%   author = {J. Guo and Y. Guo and X. Kong and R. He},
%   title = {Unsupervised Feature Selection with Ordinal Locality},
%   booktitle = {Proc. IEEE Int. Conf. Multimedia Expo (ICME)},
%   address = {Hong Kong, China},
%   pages = {1213-1218},
%   month = {Jul.},
%   year = {2017}
% }
% If you have problems, please contact us at eeguojun@outlook.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n Feature selection Method is :Unsupervised feature selection with ordinal locality \n');

if nargin < 2
    nClass = 2;
end

if nargin < 3
    para.p0 = 'sample';
    para.p1 = 1e6;
    para.p2 = 1e2;
end

% -----------------------------settings-----------------------------
X = X';
[nFea, nSmp] = size(X);
alpha = para.p1;
beta  = para.p2;
k = 5;          % number of nearest neighbors
max_Iter = 10; % maximum number of iterations
tol = 5e-7;     % the other stop criterion for iteration


%% Initialization
if strcmpi(para.p0, 'feature')
    [X,~] = mapminmax(X,1e-10,1);
    [~, L] = TripletLap(X',k);
else
    X = normcols(X);
    [~, L] = TripletLap(X,k);
    L = X*L*X';
end
Imat = eye(nFea);
Winit = Imat(:,1:nClass);
clear Imat
nPerClass = zeros(1,nClass);
Vinit = zeros(nClass,nSmp);
[label, center] = litekmeans(X'*Winit,nClass,'Replicates',10);
for j = 1:1:nClass
   nPerClass(j) = sum(label==j); 
   Vinit(j,(label==j)) = 1/sqrt(nPerClass(j));
end
Uinit = bsxfun(@times,center',sqrt(nPerClass));
valVec = [];


%% iteration
iter = 1;
while iter <= max_Iter    

    % update R
    Rii = 2*sqrt(sum(Winit.^2,2));
    Rii(Rii==0) = tol;
    R = diag(1./Rii);
    
    % update U and V
    [U,V] = solveUV(Winit'*X,Uinit,Vinit,nClass);
    
    % update W
    Temp1 = beta*R + X*X'-X*(V'*V)*X' + alpha*L;
    Temp1 = 0.5*(Temp1+Temp1');
    [eigV,eigD] = eig(Temp1);
    [~,eigI] = sort(diag(eigD),'ascend');
    W = eigV(:,eigI(1:nClass));
    
    ObjValue = sum(sum((W'*X-U*V).^2)) ...
        + alpha*trace(W'*L*W) ...
        + beta*trace(W'*R*W); 
    valVec = [valVec ObjValue];
    
    % check if stop criterion is satisfied
    leq2 = W - Winit;
    stopC = norm(leq2, 'fro'); % Frobenius norm
    if (iter==1 || mod(iter,10)==0 || stopC<tol)
        disp(['IterNo.' num2str(iter) ',stopC=' num2str(stopC,'%0.6f')]);
    end
    if stopC <= tol
        break;
    end
    
    % update init    
    Winit = W;
    Uinit = U;
    Vinit = V;
    iter = iter + 1;    
end

[~,feaIdx] = sort(sum(W.^2,2),'ascend');
rankx = feaIdx';
