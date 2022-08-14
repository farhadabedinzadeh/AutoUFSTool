function [M,J,W]=EGCFS_TNNLS(X,C,lambda,alpha,m,max_iter)
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
%% References
    % R. Zhang, Y. Zhang, and X. Li, ¡°Unsupervised Feature Selection via Adaptive Graph
    % Learning and Constraint,¡± IEEE transactions on neural networks and learning systems. 
%% Input:
    % X:the centralized data matrix d*n;
    % C:the clusternumber;
    % m:selected features;
%% Output:
    % M: the data matrix after been feature selected;
    % J: Objective function;
    % W: Projection matrix;
%% Initialization
[d,n]=size(X);
G=zeros(n,C);
for i=1:n
    g=randi([1,C]);
    G(i,g)=1;
end
 D=eye(d); 
 I=eye(n);
 U=G*((G'*G+0.001.*eye(C))^(-0.5));
%% Graph Construction
[S,~, ~] = Laplacian_CAN(X);
S=S.*(lambda./sum(S,2));
%%  
 t=1;
 err=1;
while (err > 0.1 & t<=max_iter)
    %% SOLVE W;
     temp=lambda*X*(I-U*U'-S)*X'+alpha*D;
     [W,~,~] = eig1(temp,m,0);
     clear temp;
     D=diag(1./(2*sqrt((sum(W.^2,2)+eps))));
    %% Solve U;
     temp2=X'*W*W'*X;
     [U,~,~] = eig1(temp2,C,1);
     clear temp2;
     %% Solve S;
     [S,~, gamma] = Laplacian_CAN(W'*X);
     S=lambda.*S;
     gamma=gamma/lambda;
    %% Objective function
    J(t)=trace((W'*X*(lambda*I-S-U*U')*X'*W)+alpha*W'*D*W)+trace(gamma*S'*S);
    if t>1
            err=abs(J(t)-J(t-1));
    end
    t=t+1;
end
 %% Results
    P=sqrt(sum(W.^2,2));
    [~,Q]=sort(P,'descend');
    M=X(Q(1:m,1),:);
end

%% Auxiliary function
% construct similarity matrix with probabilistic k-nearest neighbors. It is a parameter free, distance consistent similarity.
function [S,L_norm, gamma] = Laplacian_CAN(X,k, issymmetric)
% X: each column is a data point
% k: number of neighbors
% issymmetric: set W = (W+W')/2 if issymmetric=1
% W: similarity matrix

if nargin < 3
    issymmetric = 1;
end;
if nargin < 2
    k = 9;
end;

[dim, n] = size(X);
D = L2_distance_1(X, X);
[dumb, idx] = sort(D, 2); % sort each row
S = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);
    di = D(i, id);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    gamma(i)=(k*di(k+1)-sum(di(1:k))+eps)/2;
end;
gamma=mean(gamma);
if issymmetric == 1
    S = (S+S')/2;
gamma=mean(k*di(k+1)-sum(di(1:k))+eps);

Du=diag(sum(S,2));
L = Du - S; 
L_norm = Du^(-0.5)*L*Du^(-0.5);
end
end

% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function d = L2_distance_1(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b



if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);
end

% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));
% end





