function Cnew = speedUp(C)
% Refer to SCAMS: Simultaneous Clustering and Model Selection, CVPR2014 

diagmask = logical(eye(size(C,1)));
C(diagmask) = 0; %main diagonal = 0

tmp = C(:); %if 'C' is N-by-N, then 'tmp' is N*N-by-1
tmp(diagmask(:)) = []; %remove the main diagonal elements of 'C' in 'tmp'
% Then 'tmp' has a length of N*(N-1)
tmp = (tmp - min(tmp(:)))./(max(tmp(:) - min(tmp(:)))); %scale to [0,1]

affmaxo = C;
affmaxo(~diagmask) = tmp; %affmaxo(~diagmask) is a column vector
Cnew = affmaxo;

end
