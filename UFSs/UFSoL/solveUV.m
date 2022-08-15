function [U,V] = solveUV(X,Uinit,Vinit,nClass) 
% Using K-means to update U and V
%  min_U,V  ||X-U*V||_F^2
%    s.t.   V>=0, V*V'=I
% Input:
%       X       -each column is a data point
%       Uinit   -an initialization of U
%       Vinit   -an initialization of V
%       nClass  -total number of clusters
% Output:
%       U       -each column is a basis
%       V       -each column is a non-negative orthogonal coefficient

nRun = 1;
[~, nSmp] = size(X);
nPerClass = zeros(1,nClass);
V = zeros(nClass, nSmp);

for i = 1:1:nRun
    [label, center] = litekmeans(X',nClass,'Replicates',10);
    for j=1:1:nClass
       nPerClass(j) = sum(label==j); 
       V(j,(label==j)) = 1/sqrt(nPerClass(j));
    end
    U = bsxfun(@times,center',sqrt(nPerClass));
    if ( norm(X-U*V,'fro') < norm(X-Uinit*Vinit,'fro') )
        Uinit = U;
        Vinit = V;
    end
end
U = Uinit;
V = Vinit;
