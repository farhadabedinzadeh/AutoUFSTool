function D = euDist(data)

aa = sum(data.*data,2);
ab = data*data';

D = aa+aa' - 2*ab;
D(D<0) = 0;

D = sqrt(D);

D = max(D,D');