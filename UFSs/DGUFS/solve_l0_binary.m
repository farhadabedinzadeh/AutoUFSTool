function [P] = solve_l0_binary(Q, gamma)
% solve the following problem:
% min_P  ||P - Q||_F^2 + gamma*||P||_0
%  s.t.  each P_ij is in [0,1] or {0,1}
% gamma <= 1 : [0,1]
% gamma > 1  : {0,1}
% P and Q are matrixes


%%
P = Q;
if gamma > 1   % each P_ij is in {0,1}
    P(Q>0.5*(gamma+1)) = 1;
    P(Q<=0.5*(gamma+1)) = 0;
else           % each P_ij is in [0,1]
    P(Q>1) = 1;
    P(Q<sqrt(gamma)) = 0;
end        

end