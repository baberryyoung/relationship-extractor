import os,sys
import csv
import pandas as pd
import helper
dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir)
source_file = 'rawdatarela_train.txt'
with open(source_file) as infile:
    text = infile.read()
datasets = list(line.strip().split('\t') for line in text.splitlines())
rela_labels =  ['negation_of','characteristic_of','image_feature_of','result_of','location_of','before','at','during','after','num_of','size_of','subject_of','body_location_of','others']
for rela_label in rela_labels:
    count = 0
    temp_datasets = datasets
    with open(os.path.join(dir,'svmtrainset',rela_label + '_p.txt'), 'a') as outfile:
        for data in temp_datasets:
            if data[0] != rela_label:
                label = 'other'
            else:
                label = data[0]
                count += 1
            text = label
            for i in range(1,len(data)):
                text += '\t'
                text += str(data[i])
            text += '\n'
            outfile.write(text)
        outfile.close()
        print("%s is %d in all %d labels"%(rela_label,count,len(datasets)))

