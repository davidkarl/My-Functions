import klepto
from klepto.archives import file_archive


def remove_variables_from_klepto_file(file_name,variable_names_list):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name);

    if type(variable_names_list) == str:
        #input is a string:
        k = variable_names_list;
        db.archive.pop(k);
    else:
        #input is a list:
        for k in variable_names_list:
            db.archive.pop(k)









