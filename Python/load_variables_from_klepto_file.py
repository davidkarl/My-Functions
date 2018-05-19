import klepto
from klepto.archives import file_archive


def load_variables_from_klepto_file(file_name):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name);
    db.load();
    db_keys_list = list(db.keys());
    db_values_list = list(db.values());
    for k,v in zip(db_keys_list,db_values_list):
        exec( str(k) + '=' + str(v));











