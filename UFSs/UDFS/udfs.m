function [ranking] = udfs(X,nClass)
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
% run UDFS feature selection algorithm
% REF:
% @inproceedings{Yang:2011:LNR:2283516.2283660,
%  author = {Yang, Yi and Shen, Heng Tao and Ma, Zhigang and Huang, Zi and Zhou, Xiaofang},
%  title = {L2,1-norm Regularized Discriminative Feature Selection for Unsupervised Learning},
%  booktitle = {Proceedings of the Twenty-Second International Joint Conference on Artificial Intelligence - Volume Volume Two},
%  series = {IJCAI'11},
%  year = {2011},
%  isbn = {978-1-57735-514-4},
%  location = {Barcelona, Catalonia, Spain},
%  pages = {1589--1594},
%  numpages = {6},
%  url = {http://dx.doi.org/10.5591/978-1-57735-516-8/IJCAI11-267},
%  doi = {10.5591/978-1-57735-516-8/IJCAI11-267},
%  acmid = {2283660},
%  publisher = {AAAI Press},
% } 

if nargin < 2
    nClass = 2;
end

%======================setup===========================
gammaCandi = 10.^(-5);
lamdaCandi = 10.^(-5);
knnCandi = 1;
paramCell = fs_unsup_udfs_build_param(knnCandi, gammaCandi, lamdaCandi);
%======================================================

% disp('UDFS: Regularized Discriminative Feature Selection for Unsupervised Learning');
param = paramCell{1};
L = LocalDisAna(X', param);
A = X'*L*X;
W = fs_unsup_udfs(A, nClass, param.gamma);
[~, ranking] = sort(sum(W.*W,2),'descend');
ranking = ranking';

end