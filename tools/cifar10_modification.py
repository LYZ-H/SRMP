'''Modify Train CIFAR10 for PyTorch.'''
from __future__ import print_function

import numpy as np
import os
import argparse
import sys
import glob
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from zipfile import ZipFile
import argparse


def remove_index(start_index,step_num):
   filepath =  './data/Original'
   file_names = glob.glob(filepath+"/data_batch_*")
   print('Get data by index, start index: {0}; step size: {1}'.format(start_index,step_num))
   for file_name in file_names:
       f_name = file_name.split('/')
       f_name = f_name[-1]
       c_dict = {}
       with open(file_name,'rb') as fo:
            if sys.version_info[0] == 2:
               c_dict = pickle.load(fo)
            else:
               c_dict = pickle.load(fo, encoding='latin1')
       print(f_name)
       labels = c_dict['labels'] #data labels
       data = c_dict['data'] #data: a row with 3*1024 elements to store RGB 32*32 image
       batch_label = c_dict['batch_label'] # string: training batch 1-5 of 5
       filenames = c_dict['filenames']# image file name
      
       labels=labels[start_index::step_num]
       print(len(labels))
       data=data[start_index::step_num]
       print(len(data))
       filenames=filenames[start_index::step_num]
       print(len(filenames))
       c_dict['labels'] = labels
       c_dict['data'] = data
       c_dict['batch_label'] = batch_label
       c_dict['filenames'] = filenames

       outfile = './data/cifar-10-batches-py/'+f_name
    
       with open(outfile,'wb') as fout:
            c_dict = pickle.dump(c_dict,fout)
    
       with ZipFile('cifar-10-python.tar.gz', 'w') as myzip:
            zipnames = glob.glob('./data/cifar-10-batches-py/*')
            for zipname in zipnames:
                myzip.write(zipname)   


def remove_labels(rm_labels):
    filepath =  './data/Original'
    file_names = glob.glob(filepath+"/data_batch_*")
    print('==>remove_labels: '+str(rm_labels))
    for file_name in file_names:
        f_name = file_name.split('/')
        f_name = f_name[-1]
        c_dict = {}
        with open(file_name,'rb') as fo:
             if sys.version_info[0] == 2:
                c_dict = pickle.load(fo)
             else:
                c_dict = pickle.load(fo, encoding='latin1')
        print(f_name)
        labels = c_dict['labels'] #data labels
        data = c_dict['data'] #data: a row with 3*1024 elements to store RGB 32*32 image
        batch_label = c_dict['batch_label'] # string: training batch 1-5 of 5
        filenames = c_dict['filenames']# image file name

        remove_list = []
        for i in range(0,len(labels)):
            temp = labels[i]
            if temp in rm_labels:
               remove_list.append(i)
    
        labels=np.delete(labels,remove_list)
        print(len(labels))
        data=np.delete(data,remove_list,0)
        print(len(data))
        filenames=np.delete(filenames,remove_list)
        print(len(filenames))
        c_dict['labels'] = labels
        c_dict['data'] = data
        c_dict['batch_label'] = batch_label
        c_dict['filenames'] = filenames

        for i in range(0,len(labels)):
            temp = c_dict['labels'][i]
            if temp in rm_labels:
               print('===>remove failed')
               print(temp)
    
        outfile = './data/cifar-10-batches-py/'+f_name
    
        with open(outfile,'wb') as fout:
             c_dict = pickle.dump(c_dict,fout)
    
        with ZipFile('cifar-10-python.tar.gz', 'w') as myzip:
             zipnames = glob.glob('./data/cifar-10-batches-py/*')
             for zipname in zipnames:
                 myzip.write(zipname)


