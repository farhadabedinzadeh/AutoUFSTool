function  ranking  = Spectrum_graph( X, style )
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
%function [ wFeat, SF ] = fsSpectrum( W, X, style, spec )
%   Select feature using the spectrum information of the graph laplacian
%   X - the input data, each row is an instance
%   style - -1, use all, 0, use all except the 1st. k, use first k except 1st.
%   spec - the spectral function to modify the eigen values.
% --------------------------------------------------------------------
if nargin < 2
    style = -1;
end

spec = @EQU;

% --------------------------------------------------------------------
data = X';
num = size( data, 2 );
k = 5;
distX = L2_distance_1( data, data );
[ distX1, idx ] = sort( distX, 2 );
A = zeros( num );
for i = 1 : num
    di = distX1( i, 2:k+2 );
    id = idx( i, 2:k+2 );
    A( i, id ) = ( di(k+1) - di )/( k * di(k+1) - sum( di(1:k) ) + eps );
end

W = 0.5 * ( A + A' );

% --------------------------------------------------------------------
[numD,numF] = size(X);

% build the degree matrix
D = diag(sum(W,2));
% build the laplacian matrix
L = D - W;

% D1 = D^(-0.5)
d1 = (sum(W,2)).^(-0.5);
d1(isinf(d1)) = 0;

% D2 = D^(0.5)
d2 = (sum(W,2)).^0.5;
v = diag(d2)*ones(numD,1);
v = v/norm(v);

%  build the normalized laplacian matrix hatW = diag(d1)*W*diag(d1)
hatL = repmat(d1,1,numD).*L.*repmat(d1',numD,1);

% calculate and construct spectral information
[V, EVA,] = svd(hatL,'econ');
EVA = diag(EVA);
EVA = spec(EVA);

% begin to select features
wFeat = ones(numF,1)*1000;

for i = 1:numF
    f = X(:,i);
    hatF = diag(d2)*f;
    l = norm(hatF);

    if l < 100*eps
        wFeat(i) = 1000;
        continue;
    else
        hatF = hatF/l;
    end

    a = hatF'*V;
    a = a.*a;
    a = a';

    switch style
        case -1 % use f'Lf formulation
            wFeat(i) = sum(a.*EVA);
        case 0 % using all eigenvalues except the 1st
            a(numD) = [];
            wFeat(i) = sum(a.*EVA(1:numD-1))/(1-(hatF'*v)^2);
        otherwise
            a(numD) = [];
            a(1:numD-style) = [];
            wFeat(i) = sum(a.*(2-EVA(numD-style+1:numD-1)));
    end
end

% SF = 1:numD;

if style ~= -1 && style ~= 0
    wFeat(wFeat==1000) = -1000;
end
[~, ranking] = sort( wFeat, 'ascend' );
ranking = ranking';

function [newd] = EQU(d)
newd = d;