function [P] = solve_rank_lagrange(A, eta)
% solve the following problem:
% min_P  ||P - A||_F^2 + eta*rank(P)
%  s.t.  P is symmetric and positive semi-definite


%%
A = 0.5*(A+A'); % guarantee symmetry
[tempV,tempD] = eig(A);
tmpD = diag(tempD);
tmpD(tmpD<=sqrt(eta)) = 0; % eta*rank(P)
tempD = diag(tmpD);
P = tempV*tempD*tempV';   

end