%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for the Auto-UFSTool, which is an Automatic Unspervised %
% Feature Selection Toolbox                                                %
% Version 1.0 August 2022   |  Copyright (c) 2022   | All rights reserved  %
%        ------------------------------------------------------            %
% Redistribution and use in source and binary forms, with or without       %
% modification, are permitted provided that the following conditions are   %
% met:                                                                     %
%    * Redistributions of source code must retain the above copyright      %
%    * Redistributions in binary form must reproduce the above copyright.  %
%         ------------------------------------------------------           %
% "Auto-UFSTool,                                                           %
%  An Automatic MATLAB Toolbox for Unsupervised Feature Selection"         %
%         ------------------------------------------------------           %
%                    Farhad Abedinzadeh | Yegane Modaresnia                %
%          farhaad.abedinzade@gmail.com | y.modaresnia@yahoo.com           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%% The fuzzy mutual information-based unsupervised feature selection (FIMUFS) algorithm. 
%%% Please refer to the following papers: 
%%% Yuan Zhong, Chen Hongmei, Zhang Pengfei, Wan Jihong and Li Tianrui. A Novel Unsupervised 
%%% Approach to Heterogeneous Feature Selection Based on Fuzzy Mutual Information[J]. 
%%% IEEE Transactions on Fuzzy Systems, 2021.
%%% Uploaded by Yuan Zhong on Sep. 29, 2021. E-mail:yuanzhong2799@foxmail.com. 

function select_feature=ufs_FMI(data,lammda)

%%input
% data is data matrix, where rows for objects and columns for attributes without decision attribute. 
% Numerical attributes should be normalized into [0,1].
% lammda is used to adjust fuzzy neighborhood radius
%%%output
% a feature sequence for feture significance



[row, attrinu]=size(data);

delta=zeros(1,attrinu);  
for j=1:attrinu
    if min(data(:,j))==0&&max(data(:,j))==1
     delta(j)=std(data(:,j),1)/lammda; 
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%Compute the relation matrix%%%%%%%%%%%%%%%%%%%%%%%%%
 for i=1:attrinu
     col=i;
     r=[];
     eval(['ssr' num2str(col) '=[];']);
      for j=1:row      
          a=data(j,col);
          x=data(:,col);       
          for m=1:length(x)
              r(j,m)=fmiufs_kersim(a,x(m),delta(i));
          end
      end
    eval(['ssr' num2str(col) '=r;']);
 end      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%UFS based on fuzzy mutual information%%%%%%%%%%%
unSelect_Fea=[];
Select_Fea=[];
sig=[];
base=ones(row);

E=zeros(1,attrinu);
Joint_E=zeros(attrinu,attrinu);
MI=zeros(attrinu,attrinu);

for j=1:attrinu
     r=eval(['ssr' num2str(j)]);
     E(j)=entropy(r);
end

for i=1:attrinu
       ri=eval(['ssr' num2str(i)]);
    for j=1:i
        rj=eval(['ssr' num2str(j)]);
        Joint_E(i,j)=entropy(min(ri,rj));
        Joint_E(j,i)=Joint_E(i,j);
        MI(i,j)=E(i)+E(j)-Joint_E(i,j);
        MI(j,i)=MI(i,j);
    end
end

Ave_MI=mean(MI,1);

[x1,n1]=sort(Ave_MI,'descend');
sig=[sig x1(1)];
Select_Fea=n1(1);
unSelect_Fea=n1(2:end);

while ~isempty(unSelect_Fea)
    Red=[];
    for i=1:length(unSelect_Fea)
        for j=1:length(Select_Fea)
         Red(i,j)=Ave_MI(Select_Fea(j))-(Joint_E(Select_Fea(j),unSelect_Fea(i))...
             -E(unSelect_Fea(i)))/E(Select_Fea(j))*Ave_MI(Select_Fea(j));
        end
    end
        [max_sig,max_tem]=max(Ave_MI(unSelect_Fea)'-mean(Red,2));
        sig=[sig max_sig];
        Select_Fea=[Select_Fea unSelect_Fea(max_tem)];
        unSelect_Fea=setdiff(unSelect_Fea,unSelect_Fea(max_tem));
end
select_feature=Select_Fea;

function kersim=fmiufs_kersim(a,x,e)
if abs(a-x)>e
    kersim=0;
else
    if (e==0)
        if (a==x)
            kersim=1;
        else
            kersim=0;
        end
    else
        kersim=1-abs(a-x);    
    end
end

function [S]=entropy(M)
[a,b]=size(M);
K=0;
for i=1:a
    Si=-(1/a)*log2(sum(M(i,:))/a);
    K=K+Si;
    Si=0;
end
S=K;
