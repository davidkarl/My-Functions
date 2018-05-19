function Outfiles=search_files(baseDir,searchExpression,flag_recursive,flag_return_full_path)
% OUTFILES = RECURSDIR(BASEDIRECTORY,SEARCHEXPRESSION)
% A recursive search to find files that match the search expression

%initialize Outfiles:
Outfiles = {};

%get all folder in current directory
if flag_recursive==1
    d = dir(baseDir);
    isub = [d(:).isdir]; %# returns logical vector
    folder_names = {d(isub).name}';
    folder_names(ismember(folder_names,{'.','..'})) = [];
end 

%get all .m files which contain the searchExpression:
% relevant_files = dir(strcat(baseDir,'\*',searchExpression,'*.m'));
relevant_files = dir(fullfile(baseDir,searchExpression));

%put all relevant .m files in the current directory in the Outfiles
for k=1:length(relevant_files)
    if flag_return_full_path==1
        Outfiles{length(Outfiles)+1} = fullfile(baseDir,relevant_files(k).name); 
    else
        Outfiles{length(Outfiles)+1} = relevant_files(k).name;
    end
end 

%loop over folders in current directory and search for more .m files:
if flag_recursive==1
    for k=1:length(folder_names)
       new_fullfile_name = fullfile(baseDir,folder_names(k));
       OutfilesTemp = search_files(new_fullfile_name{1},searchExpression,flag_recursive,flag_return_full_path);
       if ~isempty(OutfilesTemp)
           Outfiles( (length(Outfiles)+1) : (length(Outfiles)+length(OutfilesTemp)) ) = OutfilesTemp;
       end
    end
end

