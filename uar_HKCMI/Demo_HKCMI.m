load Example

trandata=Example;
trandata=mat2gray(trandata(:,1:2));
delta_val=0.1; 
var=0.001;

select_feature_seq=uar_HKCMI(trandata,var,delta_val)


