import klepto
from klepto.archives import file_archive

def save_variables_to_klepto_file(file_name,variables_dict):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name);
    for k,v in variables_dict:
        db[k] = v;
    db.dump();














