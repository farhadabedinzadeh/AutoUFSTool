%%
% Writed by Yanfang Liu.
% Modifed at date: 2019-11-10.
% 载入标准化的数据集，k-means来测试所选特征的聚类效果
clear;clc;
% {'warpAR10P';'TOX-171';'madelon';'Isolet1';'COIL20';'ALLAML';'arcene';'GLIOMA'}
str = {'Yale_32x32'};

n_datasets = length(str);


for i_dataset = 1:n_datasets
    %%  下载数据集
    datafile = [str{i_dataset} '.mat'];
    load (datafile);
   
    for neighbor_num = 3:2:11
       %% 种子
        rand('seed',23);
        
        X = NormalizeFea(X,1);
        [~,d] = size(X);
        %% 评价指标
        ACC = zeros(2, 10);
        NMI = zeros(2, 10);
        Time = zeros(1, 10);
        
        tempACC = zeros(30,1);
        tempNMI = zeros(30,1);
        
        nc = length(unique(Y));
        
        %% 30次
        for i = 1:30
            idx = kmeans(X,nc,'emptyaction','singleton');
            tempACC(i) = clusterAccMea(Y,idx);
            tempNMI(i) = nmi(Y,idx);
        end
               
        ACC(:,1) = [mean(tempACC),std(tempACC)];
        NMI(:,1) = [mean(tempNMI),std(tempNMI)];
        Time(1) = 0;
        
        h = waitbar(0,'Please wait...');
        for j = 2:10
            t1 = clock;
            [I,obj] = RNE_obj(X,10*j,neighbor_num);
            t2 = clock;
            Time(j) = etime(t2,t1);
            for i = 1:30
                idx = kmeans(X(:,I),nc,'emptyaction','singleton');
                tempACC(i) = clusterAccMea(Y,idx);
                tempNMI(i) = nmi(Y,idx);
            end
            waitbar(j/10,h); 
            %% 输出结果
            disp(ACC);
            disp(NMI);
            disp(Time);
        end
        close(h);
    end
end
fprintf('---- completed ----\n');