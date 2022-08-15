function e = mseDens(dataAct,dataPred)
% Compute Mean Squared Error 
%
% e = mse(dataAct,dataPred)
%
% dataAct is a column vector of actual values
% dataPred is a matrix of predictions (one per column)
%
% e is the mean squared error (ignoring NaNs) for each column of
% pred.

e=1/length(dataAct)*sum((dataAct-dataPred).^2);
