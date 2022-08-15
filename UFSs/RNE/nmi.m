%% compute Nomalized mutual information

function v = nmi(label, result)

% assert(length(label) == length(result));

label = label(:);
result = result(:);

n = length(label);
n1 = length(result);

label_unique = unique(label);
c1 = length(label_unique);
result_unique = unique(result);
c2 = length(result_unique);

% check the integrity of result
if n ~= n1
    error('The clustering result is not consistent with label.');
end


% distribution of result and label
Ml = double(repmat(label,1,c1) == repmat(label_unique',n,1));
Mr = double(repmat(result,1,c2) == repmat(result_unique',n,1));
Pl = sum(Ml)/n;
Pr = sum(Mr)/n;

% entropy of Pr and Pl
Hl = -sum(Pl .* log2( Pl + eps ));
Hr = -sum(Pr .* log2( Pr + eps ));


% joint entropy of Pr and Pl

M = Ml'*Mr/n;
Hlr = -sum(M(:) .* log2(M(:) + eps));

% mutual information
MI = Hl + Hr - Hlr;

% normalized mutual information
v = sqrt((MI/Hl)*(MI/Hr));

end



