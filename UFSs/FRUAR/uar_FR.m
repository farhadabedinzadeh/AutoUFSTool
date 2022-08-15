%%Compute reduct from numerical data, categorical data and their mixtures with
%%Fuzzy rough sets without decision.
%%Dependency is employed as the heuristic rule.
%%Please refer to the following article.
%%Yuan Zhong, Chen Hongmei, Li Tianrui, Yu Zeng, Sang Binbin, and Luo Chuan. 
%%Unsupervised attribute reduction for mixed data based on fuzzy rough sets, Information Science,2021, 572:67-87.
%%Uploaded by Yuan Zhong on May 29, 2021. E-mail:yuanzhong2799@foxmail.com.
function select_feature=uar_FR(data,lammda)


%%%input:
% Data is data matrix, where rows for samples and columns for attributes without decision. 
% Numerical attributes should be normalized into [0,1]
% Fuzzy radius delta=std£¨Data£©/lammda, lammda usually takes value in [0.1,1] with 0.1
%%%output:
% a reduct--- the set of selected attributes.
[row, attrinu]=size(data);
%%
%%%%%%%%%%%%%%%%Initializes the fuzzy radius %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
delta=zeros(1,attrinu);
for j=1:attrinu
    if min(data(:,j))==0&&max(data(:,j))==1
     delta(j)=std(data(:,j),1)/lammda; % Find the fuzzy radius of the numeric attribute
    end
end
%%
%%%%%%%%%%%%%compute fuzzy relation matrices with a single attribute%%%%%%%%%
for i=1:attrinu
    col=i;
    r=[];
    eval(['ssr' num2str(col) '=[];']);
    for j=1:row      
        a=data(j,col);
        x=data(:,col);       
        for m=1:j
            r(j,m)=fruar_kersim(a,x(m),delta(i));
            r(m,j)=r(j,m);
        end
    end
    eval(['ssr' num2str(col) '=r;']);
end      
%%
%%%%%%%%%%%%%%search reduct with a forward greedy strategy%%%%%%%%%%%%%%%%%%%%%%%
red=[];
x=0;
base=ones(row);
B=1:attrinu;
for j=attrinu:-1:1
    sig=[];
    for l_1=1:length(B)
       r2=eval(['ssr' num2str(B(l_1))]);
       r1=min(r2,base);
       for l_2=1:attrinu
           temp_SIN=zeros(row);
           r_SIN=eval(['ssr' num2str(l_2)]);
           [r_SIN_temp,~,r_SIN_ic]=unique(r_SIN,'rows');
            for l_3=1:size(r_SIN_temp,1)
                    i_tem=find(r_SIN_ic==l_3);
                    temp2_SIN=min(max(1-r1,repmat(r_SIN_temp(l_3,:),row,1)),[],2);
                    temp_SIN(i_tem,:)=repmat(temp2_SIN',length(i_tem),1);
            end
            importance_SIN=sum(max(temp_SIN,[],1));
            sig(l_1,l_2)=importance_SIN/row;
       end
    end
    [x1,n1]=max(mean(sig,2));
    x=[x;x1];
    len=length(x);
    if abs(x(len)-x(len-1))>0
        base1=eval(['ssr' num2str(B(n1))]);
        base=min(base,base1);
        red=[red;B(n1)];
        B=setdiff(B,B(n1));
    else
        break;
    end
end
if length(red)==attrinu
   select_feature=red(1:end-1);
else
   select_feature=red;
end
end
