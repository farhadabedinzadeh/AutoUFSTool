function [x ft] = EProjSimplex_new(v, k)%v:f+x disntance de fushu
% the reason to do this is that, which point should have similarity to
% current point or how many points is not sure,so we should interation to decide which point is
% near to current point and the number. this is done by using threshold
%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%

if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);%15

v0 = v-mean(v) + k/n;%v0: 1*15 v-mean(v)+1/15
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10%f is residue,it is the sum of similarity of chosen point,
                          %if it is closed enough to 0,we think the chosen points is good enough
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;%k=1
        %lambda_m is used to control sum(v1(posidx)), if it is bigger than 1,then f is positive,
        %then,lambda_m will raise to let less point be neighboor to current point; if it is litter than 1, then f is neg,
        %lambda_m will decrease to let more point be neighboor to current point.
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end;
    end;
    x = max(v1,0);

else
    x = v0;
end;