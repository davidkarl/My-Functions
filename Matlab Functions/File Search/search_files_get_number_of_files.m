function counter=search_files_get_number_of_files(baseDir,searchExpression,flag_recursive)
% OUTFILES = RECURSDIR(BASEDIRECTORY,SEARCHEXPRESSION)
% A recursive search to find files that match the search expression

%initialize Outfiles:
counter = 0;

%get all folder in current directory
if flag_recursive==1
    d = dir(baseDir);
    isub = [d(:).isdir]; %# returns logical vector
    folder_names = {d(isub).name}';
    folder_names(ismember(folder_names,{'.','..'})) = [];
end

%get all .m files which contain the searchExpression:
counter = counter + length(dir(fullfile(baseDir,searchExpression)));

%loop over folders in current directory and search for more .m files:
if flag_recursive==1
    for k=1:length(folder_names)
       new_fullfile_name = fullfile(baseDir,folder_names(k));
       counter_temp = search_files_get_number_of_files(new_fullfile_name{1},searchExpression,flag_recursive);
       if ~isempty(OutfilesTemp)
           counter = counter + counter_temp;
       end
    end
end

