function [Y,L,V,Label] = DGUFS(X,nClass,S,alpha,beta,nSel) 
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Descripion: 
% This function implements the whole algorithm of our paper
% Jun Guo and Wenwu Zhu, "Dependence Guided Unsupervised 
% Feature Selection," In AAAI 2018.
%   -Source code version 1.0  2018/03/01 by Jun Guo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%       X       -each column is an original sample
%       nClass  -the number of clusters
%       S       -similarity matrix of data X
%       alpha	-regularization parameter: alpha*rank(L)
%       beta	-regularization parameter: beta*tr(S'*L)
%       nSel	-the dimension of selected features
% Output:
%       Y   	-the selected features, each column is a sample
%       L       -kernel matrix L = V'*V
%       V       -each column is a one-hot label
%       Label   -a label vecter whose size is nSmp*1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Please cite our work if you find the code helpful.
% @inproceedings{JunGuo_AAAI_2018_DGUFS,
%   author = {J. Guo and W. Zhu},
%   title = {Dependence Guided Unsupervised Feature Selection},
%   booktitle = {Proc. AAAI Conf. Artificial Intell. (AAAI)},
%   address = {New Orleans, Louisiana},
%   pages = {2232-2239},
%   month = {Feb.},
%   year = {2018}
% }
% If you have problems, please contact us at eeguojun@outlook.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% settings
addpath(genpath('.\files'));
[nFea, nSmp] = size(X);
if nSel > nFea
    error('The number of selected features is error!');
end 
H = eye(nSmp)-ones(nSmp,1)*ones(1,nSmp)/nSmp; % H=H'
H = H/(nSmp-1); 
% (nSmp-1)^2 may be more effective than (nSmp-1)^1
Y = zeros(nFea,nSmp);
Z = zeros(nFea,nSmp);
M = zeros(nSmp,nSmp);
L = zeros(nSmp,nSmp);
Lamda1 = zeros(nFea,nSmp);
Lamda2 = zeros(nSmp,nSmp);
rho = 1.1;
max_mu = 1e10;
mu = 1e-6;
max_Iter = 5; % maximum number of iterations
tol = 5e-7;     % the other stop criterion for iteration


%% iteration

iter = 1;
while iter <= max_Iter
    
    % update Z
    temp1 = X - Y - ((1-beta)*Y*H*L*H-Lamda1)/mu;
    Z = X - solve_l20(temp1,(nFea-nSel));  
    
    % update Y
    temp1 = Z + ((1-beta)*Z*H*L*H+Lamda1)/mu;
    Y = solve_l20(temp1,nSel); 

    % update L
    temp2 = ((1-beta)*speedUp(H*Y'*Z*H) + beta*S - Lamda2)/mu + M;
    L = solve_rank_lagrange(speedUp(temp2), 2*alpha/mu);
    
    % update M
    M = L + Lamda2/mu;
    gamma = 5e-3;
    M = solve_l0_binary(M, 2*gamma/mu);    
    M = M - diag(diag(M)) + eye(nSmp);
                    
    % check if stop criterion is satisfied
    leq1 = Z - Y;
    leq2 = L - M;
    stopC1 = max(max(abs(leq1))); % Infinite norm
    stopC2 = max(max(abs(leq2))); % Infinite norm
    if (iter==1 || mod(iter,2)==0 || ((stopC1<tol)&&(stopC2<tol)) )
        disp(['Iter ' num2str(iter) ',mu=' num2str(mu,'%2.4e') ...
        ',stopC= ' num2str(stopC1,'%2.4e') ', ' num2str(stopC2,'%2.4e')]);
    end
    if (stopC1<tol)&&(stopC2<tol)
        break;
    end
    
    % update Lagrange multipliers    
    Lamda1 = Lamda1 + mu*leq1; %(Z-Y);
    Lamda2 = Lamda2 + mu*leq2; %(L-M);
    mu = min(max_mu, mu*rho);

    iter = iter + 1;
    
end


%% obtain labels
[eigV,eigD] = eigs(max(L,L'),nClass,'la');
V = eigV*sqrt(eigD); %[nSmp,nClass]=size(V)
[~, Label] = max(abs(V),[],2); 
V = V'; % final: [nClass,nSmp]=size(V)

end
