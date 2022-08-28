function [P] = solve_l20(Q, m)
% solve the following problem:
% min_P  ||P - Q||_F^2
%  s.t.  ||P||_2,0 <= m

%%
b = sum(Q.^2,2); % b(i) is the (l2-norm)^2 of the i-th row of Q
[~,idx] = sort(b,'descend');
P = zeros(size(Q));
P(idx(1:m),:) = Q(idx(1:m),:);

end