function Selection_Method = UFSNaming()
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

cprintf('*black',  ' =========== Auto-UFSTool - An Automatic MATLAB Toolbox for Unsupervised Feature Selection =========== \n');
fprintf('\n');


load('UFS_Names.mat','UFS_Names','listUFS')


[ methodID ] = SelectMethod( UFS_Names );

if (methodID >= 1) && (methodID < 24)
    Selection_Method = listUFS{methodID}; % Selected
else
    cprintf('*red',  ' Warning! Please Check the Method and Try again. \n');
    Selection_Method =[];
end
end

