%%Computing reduct from hybrid data in fuzzy information system.
%%Fuzzy complementary mutual information is employed as the heuristic rule.
%%Please refer to the following article.
%%Yuan Zhong, Chen Hongmei, Yang Xiaoling, Li Tianrui, and Liu Keyu. 
%%Fuzzy complementary entropy using hybrid-kernel function and its
%%unsupervised attribute reduction, Knowledge-Based Systems, 2021, 231: 107398.
%%Uploaded by Yuan Zhong on Dec 30, 2021. E-mail:yuanzhong2799@foxmail.com.   
function select_feature=uar_HKCMI(data,var,delta_val)
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
%%input
% data is data matrix without decision, where rows for objects and columns for attributes. 
% Numerical attributes should be normalized into [0,1]
% delta_val denotes Gaussian parameter delta [0,1] with step size 0.1.
% var is stop criterion.
%%%output
% a feature sequence for feture significance



[row, attrinu]=size(data);

delta=zeros(1,attrinu);%Initialize the radius 
for j=1:attrinu
    if min(data(:,j))==0&&max(data(:,j))==1 %Find the radius of the numeric feature
     delta(j)=delta_val;
    end
end
%%%%%%%%%%%%%Compute the relation matrix %%%%%%%%%
 for i=1:attrinu
     col=i;
     r=[];
     eval(['ssr' num2str(col) '=[];']);
      for j=1:row  
          a=data(j,col);
          x=data(:,col);     
          for m=1:j
              r(j,m)=hkuar_kersim(a,x(m),delta(i));
              r(m,j)=r(j,m);
          end
      end
    eval(['ssr' num2str(col) '=r;']);
 end

%%%%%%%%%%%data reduct based on fuzzy complement mutual entropy%%%%%%%%%%%%
red=[];
x=0;
base=ones(row);
B=1:attrinu;
for j=attrinu:-1:1 
    sig=[];% For save significance
    for i=1:length(B)
       r1=eval(['ssr' num2str(B(i))]);
       for k=1:attrinu
        ck=eval(['ssr' num2str(k)]);
        sig(i,k)=fuzzycentropy(ck)+fuzzycentropy(r1.*base)-fuzzycentropy(r1.*(ck.*base));
       end
    end
    [x1,n1]=max(mean(sig,2));
    x=[x;x1];
    len=length(x);
    if abs(x(len)-x(len-1))>var
        base1=eval(['ssr' num2str(B(n1))]);
        base=base.*base1;%algebraic product.
        red=[red B(n1)];
        B=setdiff(B,B(n1));
    else
        break
    end
end
if length(red)==attrinu
select_feature=red(1:end-1);
else
    select_feature=red;
end
end
%%%%%Calculate hybrid-kernel fuzzy similaity %%%%%%%
function kersim=hkuar_kersim(a,x,e)
if e==0
   if (a==x)
            kersim=1;
        else
            kersim=0;
   end
else
       kersim=exp(-(a-x)^2/(2*e^2)); 
end
end
%%%%%Calculate the fuzzy complementary entropy of relational matrix %%%%%%%
function [S]=fuzzycentropy(M)
[a,b]=size(M);
K=0;
for i=1:a
    Si=(1/a)*(1-sum(M(i,:))/a);
    K=K+Si;
    Si=0;
end
S=K;
end
