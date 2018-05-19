import glob
import os
import fnmatch
#from path import path
import sys 
from fnmatch import filter
from functools import partial
from itertools import chain
from os import path, walk
import pathlib


def search_file(pattern='*', file_directory = os.getcwd()):
    return [os.path.basename(c) for c in glob.glob(file_directory + '/' + pattern)]



##Change working directory if wanted:
#new_file_directory = 'blabla';
#os.chdr(new_file_directory);
#
##Search files ending with .txt extentions:
#files_in_os_working_directory = glob.gob("*.txt");
##Search for ALL files in directory:
#files_in_os_working_directory = os.listdir();
#files_in_os_working_directory = os.listdir(os.getcwd());
#files_in_os_working_directory = os.listdir(new_file_directory);
#for file in files_in_os_working_directory:
#    print(file);
#
#
##Get root, directories and files in directory:
#for root, dirs, files in os.walk("/mydir"):
#    for file in files:
#        if file.endswith(".txt"):
#             print(os.path.join(root, file))
#
#for root, dirs, files in os.walk(dir):
#    for f in files:
#        fullpath = os.path.join(root, f)
#        if os.path.splitext(fullpath)[1] == '.txt':
#            print(fullpath)
#            
#
#all_txt_files = filter(lambda x: x.endswith('.txt'), os.listdir(new_file_directory))            
#
#filenames_without_extension = [os.path.basename(c).split('.')[0:1][0] for c in glob.glob('your/files/dir/*.txt')]
#filenames_with_extension = [os.path.basename(c) for c in glob.glob('your/files/dir/*.txt')]
#
#def locate(pattern, root=os.curdir):
#    '''Locate all files matching supplied filename pattern in and below
#    supplied root directory.'''
#    for path, dirs, files in os.walk(os.path.abspath(root)):
#        for filename in fnmatch.filter(files, pattern):
#            yield os.path.join(path, filename)


        
        
        
        