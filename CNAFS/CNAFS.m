function [W,score,index, objectives]=CNAFS(X, c, alpha, beta, lambda, gamma, epsilon, NITER, NMF_K)
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
%Reference: [1] Aihong Yuan, Mengbo You*, Dongjian He and Xuelong Li, 
%         "Convex Non-Negative Matrix Factorization With Adaptive Graph for Unsupervised Feature Selection,"
%         in IEEE Transactions on Cybernetics, doi: 10.1109/TCYB.2020.3034462, 2020
%
%Input:
%      X: d by n matrix, n samples with d dimensions.
%      c: the desired cluster number.
%      alpha, beta, lambda, gamma, epsilon: parameters refer to paper.
%      NITER: the desired number of iteration.
%      NMF_K: the number of basis vectors for Nonnegative Matrix Factorization
%Output:
%      W: d by c projection matrix.
%      score: d-dimensional vector, preserves the score for each dimensions.
%      indx: the sort of features for selection.
%      objectives: the vector of objective function values
[d,n]=size(X);
H=eye(n)-ones(n,n)./n;
F = orth(rand(n,c));
W = rand(d,c);
G = abs(rand(n, NMF_K));
V = abs(rand(NMF_K, n));
diff=1; iteration=1;
Q = ones(NMF_K, NMF_K)-eye(NMF_K);
while (diff > 0.1 && iteration< NITER)
    D_weight = diag( 0.5./sqrt(sum(W.*W,2)+epsilon));
    for i=1:n
        for j=1:n
            S(i,j)=exp(-(alpha* norm(F(i,:)-F(j,:),2)^2 + 0.5*gamma* norm(V(:, i)-V(:, j),2)^2)/(2*alpha*beta));
        end
        S(i,:)=S(i,:)./sum(S(i,:));
    end
    S=(S+S')./2;
    D=diag(sum(S,2));
    L=D-S;
    H_centering=eye(n)-ones(n,1)*ones(1,n)./n;
    W=real(inv(X*H_centering*X'+lambda.*D_weight)*X*H_centering*F);
    
    A=H+2*alpha.*L;
    B=H*X'*W;
    F=gpi(A,B,1);%refer to the implementation for paper "A generalized power iteration method
    %for solving quadratic problem on the Stiefel manifold", Feiping Nie, Rui Zhang, and Xuelong Li, 2017.
    infor_entropy=0;
    for pos_i=1:n
        for pos_j=1:n
            infor_entropy=infor_entropy+S(pos_i,pos_j)*log(S(pos_i,pos_j)+1e-10);
        end
    end
    
    G = G .* (X'*X*V'./ (X'*X*G*(V*V')));
    V = V .* ((G'*(X'*X)+gamma*V*S) ./ (G'*(X'*X)*G*V+gamma*V*D+epsilon*Q*V));
    
    %objective function value
    objectives(iteration) = norm(X-X*G*V, 'fro')^2 + norm(H*(X'*W-F),'fro')^2+lambda*trace(W'*D_weight*W)+2*alpha*(trace(F'*L*F)+beta*infor_entropy) + gamma*trace(V*L*V') + epsilon*trace(V'*Q*V);
    if iteration>1
        diff = abs(objectives(iteration-1)-objectives(iteration));
    end
    iteration = iteration+1;
end

score=sum((W.*W),2);
[~,index]=sort(score,'descend');
end
