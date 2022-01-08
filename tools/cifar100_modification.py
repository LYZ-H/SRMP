'''Modify Train CIFAR100 for PyTorch.'''
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


def remove_index(start_indices,step_num):
   filepath =  './data/cifar-100-python'
   file_names = glob.glob(filepath+"/train*")
   print('Get data by index, start index: {0}; step size: {1}'.format(start_indices,step_num))
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
  
       old_coarse_labels = c_dict['coarse_labels']#data coarse labels
       old_fine_labels = c_dict['fine_labels']#data coarse labels
       old_data = c_dict['data']#data: a row with 3*1024 elements to store RGB 32*32 image
       batch_label = c_dict['batch_label'] # string: training batch 1-5 of 5
       old_filenames = c_dict['filenames']# image file name

       coarse_labels = []
       fine_labels = []
       data = []
       filenames = []
       for i in range(0,len(start_indices)):
           s_index = start_indices[i]
           coarse_labels.extend(old_coarse_labels[s_index::step_num])
           fine_labels.extend(old_fine_labels[s_index::step_num])
           data.append(old_data[s_index::step_num])
           filenames.extend(old_filenames[s_index::step_num])
       data = np.vstack(data)

       print('modified/original lenght: {0}/{1}'.format(len(fine_labels),len(old_fine_labels)))
       #print('modified/original lenght: {0}/{1}'.format(len(coarse_labels),len(old_coarse_labels)))

       #print('data=>modified/original lenght: {0}/{1}'.format(len(data),len(old_data)))

       #print('filename=>modified/original lenght: {0}/{1}'.format(len(filenames),len(old_filenames)))

       c_dict['coarse_labels'] = coarse_labels
       c_dict['fine_labels'] = fine_labels
       c_dict['data'] = data
       c_dict['batch_label'] = batch_label
       c_dict['filenames'] = filenames

       outfile = './data/cifar-100-python/'+f_name
    
       with open(outfile,'wb') as fout:
            c_dict = pickle.dump(c_dict,fout)
    
       '''
       with ZipFile('cifar-10-python.tar.gz', 'w') as myzip:
            zipnames = glob.glob('./data/cifar-10-batches-py/*')
            for zipname in zipnames:
                myzip.write(zipname)   
       '''


def remove_labels(rm_labels):
    pass

