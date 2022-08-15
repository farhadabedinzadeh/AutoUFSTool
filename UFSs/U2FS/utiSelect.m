function feats_selected = utiSelect(data, alphas, numFeats2select, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  utiSelect - utility metric subset selection
%
%  A function which selects the best N channels of A for the least-squared
%  problem min_{x} 1/2 * ||Ax - b ||^2 which best estimates b.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  INPUTS:
%       data                - Data matrix (N x d). N is the number of
%                             samples and d the number of features/dimensions.
%
%       alphas              - set of highest eigenvectors (E in the paper)
%                             low-dimensional representation of the data.
%                             (N x c). N is the number of samples and c is
%                             the number of clusters/reduced dimensions
%
%       numFeats2select     - Number of features to be selected. Can be
%                             single number or vector, if several
%                             'numFeats2select' are to be evaluated.
%
%  OUTPUTS:
%       feats_selected      - Indices of features selected
%
%
%  Abhijith Mundanad Narayanan - abhijith@kuleuven.be
%  KU Leuven
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Default conditions
noflags = 1;
if(nargin<3)
    numFeats2select = size(data,2);
end

if nargin > 3
    for i = 1:2:length(varargin)
        Param = varargin{i};
        Value = varargin{i+1};
        if ~isstr(Param)
            error('Flag arguments must be strings')
        end
        Param = lower(Param);
        switch Param
            case 'method'
                method = Value;
            case 'lags'
                noflags = Value+1;
        end
    end
end


% Calculate the auto and crosscovariances
RXX = (data'*data)/size(data,1);
RXY = (data'*alphas)/size(data,1);
no_of_channels = size(RXX,2)/noflags;

% Initialise a list of original indices/channels
chnl_list = (1:no_of_channels)';

% Check if normal utility or minimum norm utility definition
% to be used based on rank of covariance
rankMat = rank(RXX);
if(rankMat<size(RXX,2))
    min_norm_flag = 1;
else
    min_norm_flag = 0;
end

% Indicator matrices to book-keep removed channels

% selector of columns (channels and lags)
col_sel = ones(1,size(RXX,2));

% indicator vector for channels
node_ids = ones(1,no_of_channels);
node_ids = logical(node_ids);

% Vector to store removed channels
deleted_channels = zeros(no_of_channels,1);
del_ch_count = 0;

%% Adaptation to code to be efficient in experiments

if length(numFeats2select)==1 % ******************************
    while(del_ch_count+1 <= (no_of_channels-numFeats2select))
        
        % Populate a list of remaining channel numbers
        temp_chnl_list = chnl_list(node_ids,:);
        X_sel = RXX(logical(col_sel), logical(col_sel));
        RXY_sel = RXY(logical(col_sel),:);
        
        util = zeros(size(temp_chnl_list,1),1);
        
        eigvals = eigs(X_sel);
        lambda_scaling = min(eigvals(eigvals>0));
        
        %%%%%%%% Checking rank
        if rankMat == size(X_sel,1)
            min_norm_flag = 0;
        end
        %%%%%%%
        
        % Recursive computation of inverse - reduce complexity
        % Only after the first inverse has been computed
        % ***********************************
        if(del_ch_count)
            k = idx;
            S = Xinv((k-1)*noflags+1:k*noflags,(k-1)*noflags+1:k*noflags);
            V1 =  Xinv((k-1)*noflags+1:k*noflags,1:(k-1)*noflags);
            V2 =  Xinv((k-1)*noflags+1:k*noflags,(k*noflags)+1:end);
            V = [V1,V2];
            C = [];
            C1 = Xinv(1:(k-1)*noflags,1:(k-1)*noflags);
            C = [C1,Xinv(1:(k-1)*noflags,(k*noflags)+1:end)];
            C2 = Xinv((k*noflags)+1:end,1:(k-1)*noflags);
            C = [C;C2,Xinv((k*noflags)+1:end,(k*noflags)+1:end)];
            Xinvnew = C- V'*(S\V);
            Xinv = Xinvnew;
        else
            lambda_I = (lambda_scaling*1.0e-5)*eye(size(X_sel,1));
            Xinv = (X_sel + min_norm_flag*lambda_I)\eye(size(X_sel,1));
        end
        % ************************************************************
        % Compute the new decoder with remaining channels
        W = Xinv * RXY_sel;
        
        % Compute utility of all remaining channels
        for k = 1:size(temp_chnl_list,1)
            % Select decoder weights of channel k and its lags
            Wkq = W((k-1)*noflags+1:k*noflags,:);
            % utility computation of channel k ( if block utility: channel and its lags)
            S = Xinv((k-1)*noflags+1:k*noflags,(k-1)*noflags+1:k*noflags);
            util(k) = sum(diag(Wkq'*(S\Wkq)));
        end
        
        % Find the index of channel with least utility
        [~, idx] = min(util);
        
        % Pick the actual channel number of least utility
        % from list of remaining channel numbers
        temp_ch_sel = temp_chnl_list(idx, :);
        
        % Delete that channel from the set
        row_id = find(chnl_list==temp_ch_sel);
        col_sel((row_id-1)*noflags+1:row_id*noflags) = 0;
        node_ids(row_id) = false;
        
        % store the deleted/removed channel
        del_ch_count = del_ch_count+1;
        deleted_channels(del_ch_count,1) = temp_ch_sel;
        
    end
    
    feats_selected = chnl_list(node_ids,:);
    
    % Best N channels in ascending order of significance
    % i.e. the last deleted channel comes first
    %         ch_selected = flipud(deleted_channels);
    %         ch_selected = ch_selected(1:N);
    %%
else % ***********************************
    % Order numfeats to extract from higher to lower
    numFeats = sort(numFeats2select,'descend');
    lf = 1;
    feats_selected = cell(length(numFeats),1);
    while(del_ch_count+1 <= (no_of_channels-numFeats(lf)))
        
        % Populate a list of remaining channel numbers
        temp_chnl_list = chnl_list(node_ids,:);
        X_sel = RXX(logical(col_sel), logical(col_sel));
        RXY_sel = RXY(logical(col_sel),:);
        
        util = zeros(size(temp_chnl_list,1),1);
        
        eigvals = diag(X_sel);
        lambda_scaling = min(eigvals(eigvals>0));
        
        %%%%%%%% Checking rank
        if rankMat == size(X_sel,1)
            min_norm_flag = 0;
        end
        %%%%%%%
        % Recursive computation of inverse - reduce complexity
        % Only after the first inverse has been computed
        % ***********************************
        if(del_ch_count)
            k = idx;
            S = Xinv((k-1)*noflags+1:k*noflags,(k-1)*noflags+1:k*noflags);
            V1 =  Xinv((k-1)*noflags+1:k*noflags,1:(k-1)*noflags);
            V2 =  Xinv((k-1)*noflags+1:k*noflags,(k*noflags)+1:end);
            V = [V1,V2];
            C = [];
            C1 = Xinv(1:(k-1)*noflags,1:(k-1)*noflags);
            C = [C1,Xinv(1:(k-1)*noflags,(k*noflags)+1:end)];
            C2 = Xinv((k*noflags)+1:end,1:(k-1)*noflags);
            C = [C;C2,Xinv((k*noflags)+1:end,(k*noflags)+1:end)];
            Xinvnew = C- V'*(S\V);
            Xinv = Xinvnew;
        else
            lambda_I = (lambda_scaling*1.0e-5)*eye(size(X_sel,1));
            Xinv = (X_sel + min_norm_flag*lambda_I)\eye(size(X_sel,1));
        end
        % ************************************************************
        % Compute the new decoder with remaining channels
        W = Xinv*RXY_sel;
        
        % Compute utility of all remaining channels
        for k = 1:size(temp_chnl_list,1)
            % Select decoder weights of channel k and its lags
            Wkq = W((k-1)*noflags+1:k*noflags,:);
            % utility computation of channel k ( if block utility: channel and its lags)
            S = Xinv((k-1)*noflags+1:k*noflags,(k-1)*noflags+1:k*noflags);
            util(k) = sum(diag(Wkq'*(S\Wkq)));
        end
        
        % Find the index of channel with least utility
        [~, idx] = min(util);
        
        % Pick the actual channel number of least utility
        % from list of remaining channel numbers
        temp_ch_sel = temp_chnl_list(idx, :);
        
        % Delete that channel from the set
        row_id = find(chnl_list==temp_ch_sel);
        col_sel((row_id-1)*noflags+1:row_id*noflags) = 0;
        node_ids(row_id) = false;
        
        % store the deleted/removed channel
        del_ch_count = del_ch_count+1;
        deleted_channels(del_ch_count,1) = temp_ch_sel;
        
        if del_ch_count+1 > (no_of_channels-numFeats(lf)) && lf < length(numFeats)
            feats_selected{lf,1} = chnl_list(node_ids,:);
            lf = lf +1;
        end
    end
    
    feats_selected{lf,1} = chnl_list(node_ids,:);
end


end