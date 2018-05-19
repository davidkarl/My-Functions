import re
import traceback
import inspect

pattren = re.compile(r'[\W+\w+]*get_variable_name\((\w+)\)')
def get_variable_name(x):
    return pattren.match( traceback.extract_stack(limit=2)[0][3]) .group(1)

# def retrieve_name(var):
#         """
#         Gets the name of var. Does it from the out most frame inner-wards.
#         :param var: variable to get name from.
#         :return: string
#         """
#         for fi in reversed(inspect.stack()):
#             names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
#             if len(names) > 0:
#                 return names[0]

a = 1
b = a
c = b
print(type(get_variable_name(a)))
print(get_variable_name(b))




