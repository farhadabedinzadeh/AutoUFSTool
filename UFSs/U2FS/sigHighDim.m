 function selSig = sigHighDim(data)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  sigHighDim 
%  
%  Code for RBF kernel parameter estimation for high dimensional data. 
%  Porposed in "Utility metric for Unsupervised feature selection", 
%  Amalia Villa, Abhijith Mundanad Narayanan, Sabine Van Huffel, Alexander
%  Bertrand, Carolina Varon
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  selSig = sigHighDim(data)
%  
%  INPUTS: 
%       data         - Data matrix (N x d). N is the number of 
%                      samples and d the number of features/dimensions
%
%  OUTPUTS: 
%       selSig       - Estimated sigma
%
%
%  Amalia Villa - amalia.villagomez@kuleuven.be
%  KU Leuven
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Manhattan distances delta

% Manhattan distance per dimension
for i = 1 : size(data,2)
    x1 = data(:,i).*ones(size(data,1)); 
    x2 = data(:,i)'.*ones(size(data,1));
    % Smart calculation distance
    Nom = triu(abs(x1-x2),1);
    dn = reshape(Nom,numel(Nom),1); 
    dn = dn(dn~=0);
    if sum(sum(dn)) == 0
        delta(i)=0;
    else
        delta(i) = mean(dn); % Mean of the distance per dimension
    end
end

%% Weight vector ws

% Weight per dimension
for i = 1 : size(data,2)
    % Estimate distribution data
    [f,xi] = ksdensity(data(:,i));
    % Normalize density function
    f = f./sum(f);
    
    distFeat(i,1:length(xi)) = f;
    xax(i,:) = xi;
    
    % Fit distribution data with gaussian and calculate error
    try
        fity = fit(xax(i,:)',f','gauss1');
        f2 = feval(fity,xi);
        f2 = f2./sum(f2);
        fitthingy(i,:) = f2';
        errDist(i) = mseDens(f2,f');
    catch
        % For the cases where fit takes out an error
        errDist(i) = 1000; % very large distance
    end
end

% All errs set to 1000, adjuts them to maximum of other errors
valRep = max(errDist(errDist~=1000)); % The code requires the installation of the Curve Fitting Toolbox from Matlab.
errDist(errDist==1000) = valRep;
% Normalize all weights so sum is 1
ws = errDist/sum(errDist);

%% Final estmation sigma
selSig = sum(delta.*ws);


