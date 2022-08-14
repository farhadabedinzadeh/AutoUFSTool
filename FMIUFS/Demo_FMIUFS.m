
load Example

trandata=Example;
trandata(:,3:4)=normalize(trandata(:,3:4),'range');
lammda=1; 
feature_seq=ufs_FMI(trandata,lammda)


