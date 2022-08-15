%%  Robust neighborhood embedding for unsupervised feature selection
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
% Written by Yanfang Liu (liuyanfang003@163.com), Oct 2018
% for paper 'Robust neighborhood embedding for unsupervised feature selection'
%
%%%%%%%%%%%%% Input %%%%%%%%%%%%%%%%%%%%%%%%
% X: n x d
% m: the number of the selected features
% numNeighbor: the number of neighbors to construct manifold matrix
%
%%%%%%%%%%% Output %%%%%%%%%%%%%%%%%%%%%%%%%%
% I: the index set of the selected features


function [I,obj] = RNE_obj(X,m,numNeighbor)

[n, d] = size(X);
%% W
W = LLE(X,numNeighbor); % Get the reconstruction matrix (embedding matrix)

A = (eye(n)-W')*X;
AA = A'*A;
AAplus = 0.5*(abs(AA)+AA);
AAsubtract = 0.5*(abs(AA)-AA);
H = rand(d, m);  % initialize H
M = A*H;
Y = zeros(n,m);
mu = 1.1;
gamma = 10;
max_gamma = 1e10;
alpha = 1e+3;
iter_num = 50;
iter_numH = 30;

for iterTotal=1:iter_num
    AM = A'*M;
    AY = A'*Y;
    AMplus = 0.5*(abs(AM)+AM);
    AMsubtract = 0.5*(abs(AM)-AM);
    AYplus = 0.5*(abs(AY)+AY);
    AYsubtract = 0.5*(abs(AY)-AY);
    
    %% Update H
    for i = 1:iter_numH
        G1 = diag(sqrt(1./(diag(H'*H)+eps)));
        H = H*G1;
        H = H.*sqrt((alpha*H+gamma*AMplus+gamma*AAsubtract*H+AYplus)./(alpha*H*(H'*H)+gamma*AMsubtract+gamma*AAplus*H+AYsubtract+eps));
    end
    %% Update M
    temp = A*H-Y/gamma;
    M = wthresh(temp,'s',1/gamma);
%     M = sign(temp).*max(abs(temp) - 1/gamma,0);
    %% Update Y
    Y = Y+gamma*(M-A*H);
    %% Update gamma
    gamma=min(mu*gamma,max_gamma);
    %% obj
    obj(iterTotal) = sum(sum(abs(A*H)))+1/4*alpha*norm((H'*H-eye(m)),'fro').^2; 
%     disp(['obj:',num2str(iterTotal),': ',num2str(obj(iterTotal))]);
%     disp(' ');
end

tempVector = sum(H.^2, 2);
[~, value] = sort(tempVector, 'descend'); % sort tempVecror (H) in a descend order
I = value(1:m);
end

