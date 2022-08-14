%Title: A generalized power iteration method for solving quadratic problem on the Stiefel manifold
%% Authors: Feiping Nie, Rui Zhang, and Xuelong Li.
%Citation: SCIENCE CHINA Information Sciences 60, 112101 (2017); doi: 10.1007/s11432-016-9021-9
%View online: http://engine.scichina.com/doi/10.1007/s11432-016-9021-9
%View Table of Contents:http://engine.scichina.com/publisher/scp/journal/SCIS/60/11
%Published by the Science China Press
%% Generalized power iteration method (GPI) for solving min_{W｡ｮW=I}Tr(W｡ｯAW-2W^TB)
%Input: A as any symmetric matrix with dimension m*m; B as any skew matrix with dimension m*k,(m>=k);
%In particular, s can be chosen as 1 or 0, which stand for different ways of determining relaxation parameter alpha.
%i.e. 1 for the power method and 0 for the eigs function.
%Output: solution W and convergent curve.
function W=gpi(A,B,s)
if nargin<3
    s=1;
end
[m,k]=size(B);
if m<k
    disp('Warning: error input!!!');
    W=null(m,k);
    return;
end
A=max(A,A');

if s==0
    alpha=abs(eigs(A,1));
else if s==1
        ww=rand(m,1);
        for i=1:10
            m1=A*ww;
            q=m1./norm(m1,2);
            ww=q;
        end
        alpha=abs(ww'*A*ww);
    else
        disp('Warning: error input!!!');
        W=null(m,k);
        return;
    end
end

err1=1;t=1;
W=orth(rand(m,k));
A_til=alpha.*eye(m)-A;
while t<5
    M=A_til*W+2*B;
    [U,~,V]=svd(M,'econ');
    W=U*V';
    obj(t)=trace(W'*A*W-2.*W'*B);
    if t>=2
        err1=abs(obj(t-1)-obj(t));
    end
    t=t+1;
end
end