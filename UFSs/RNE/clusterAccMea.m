%% Measure percentage of Accuracy

function Acc = clusterAccMea(T,idx)

% Output
% Acc = Accuracy of clustering results

% Input
% T = 1xn target index
% idx =1xn matrix of the clustering results

% EX:
% X=[randn(200,2);randn(200,2)+6,;[randn(200,1)+12,randn(200,1)]]; T=[ones(200,1);ones(200,1).*2;ones(200,1).*3];
% idx=kmeans(X,3,'emptyaction','singleton','Replicates',5);
%  Acc = ClusterAccMea(T,idx)


% Step 1: get the index of each cluster of the target index
k1 = length(unique(T));
n1 = length(T);
for i = 1:k1
    temp = find(T==i);
    a{i} = temp; %#ok<AGROW>
end

% Step 2: get the index of the learned cluster
k2 = length(unique(idx));
n2 = length(idx);

if n1 ~= n2
    disp('These two indices do not match!');
end

for i = 1:k2
    temp = find(idx==i);
    b{i} = temp; %#ok<AGROW>
end


% Step 3: compute the cost matrix of these two indices
disMat = zeros(k1,k2);
for i1 = 1:k1
    for i2 = 1:k2
        disMat(i1,i2) = length(intersect(a{i1},b{i2}));
    end
end


% Step 4: compute the cluster accuracy
[~,cost] = munkres(-disMat);
Acc = -cost/n1;




