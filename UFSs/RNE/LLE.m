%% Algorithm for computing the reconstruction matrix
%
%%%%%%%%%%% Input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = data as N x D matrix (D = dimensionality, N = #points)
% K = number of neighbors

%%%%%%%%%%% Output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wf: reconstruction matrix (embedding matrix)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W_squar] = LLE(X,K)

X = X';
[D,N] = size(X);

% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
fprintf(1,'-->Finding %d nearest neighbors.\n',K);
X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X);
[~,index] = sort(distance);
neighborhood = index(2:(1+K),:);

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
fprintf(1,'-->Solving for reconstruction weights.\n');
if(K>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = zeros(K,N);
for ii=1:N
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
   W(:,ii) = C\ones(K,1);                           % solve Cw=1
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end;

% Sparse reconstruction matrix 
W_squar = sparse(repmat(1:N,K,1),neighborhood,W,N,N);
% Full representation of the sparse reconstruction matrix
% W_squar = full(W_squar);

% % STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF COST MATRIX M=(I-W)'(I-W)
% fprintf(1,'-->Computing embedding regularizer (manifold matrix).\n');
% 
% %M = eye(N,N); % use a sparse matrix with storage for 4KN nonzero elements
% M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
% for ii=1:N
%    w = W(:,ii);
%    jj = neighborhood(:,ii);
%    M(ii,jj) = M(ii,jj) - w';
%    M(jj,ii) = M(jj,ii) - w;
%    M(jj,jj) = M(jj,jj) + w*w';
% end;

end
