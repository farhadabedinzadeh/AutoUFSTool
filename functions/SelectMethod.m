function [ methodID ] = SelectMethod (list ) 
% 'Black','Cyan','Magenta','Blue','Green','Red','Yellow','White'
cprintf('*red',  ' Please Select a Feature Selection Method from the List (Write Number of Method) :\n');

fprintf('...................................................................\n');
for i=1:length(list)
   fprintf('[%d] %s \n',i,list{i});
   fprintf('...................................................................\n');

end
methodID = input('> ');


end