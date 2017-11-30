#!/usr/bin
# -*- coding=utf-8 -*-
import operator
import  os,sys
import argparse
import jieba
import math
jieba.load_userdict("mydict.txt")
dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir)
parser = argparse.ArgumentParser()
parser.add_argument("-s","--source_path", help="the path of the rawdata file",default ="rawdata")
parser.add_argument("-d","--dest_path", help="the path to save file",default="rawdata")
parser.add_argument("-o","--operation", help="the operation type of this dataset used for(train/dev/test)", default="train")
args = parser.parse_args()
filepath = os.path.join(dir,args.dest_path)
raw_data_file = os.path.join(dir,args.source_path)
def read_txt(path):
    text = ""
    try:
        with open(path + 'record.txt', encoding='utf-8') as f:
            text = f.read()
            f.close()
    except Exception:
        return None
    return text

def read_ann(path):
    with open(path + 'record.ann', encoding='utf-8') as f:
        lines = f.readlines()
        entity = dict()
        rela = dict()
        for line in lines:
            #
            if line[0] == 'T':
                [id, annotation, content] = line.strip().split('\t')
                if ';' in annotation:
                    pass
                else:
                    [entity_type, start_index, end_index] = annotation.split(' ')
                    entity[id] = [entity_type, int(start_index), int(end_index), content]
            elif line[0] =='R':
                [id, annotation] = line.strip().split('\t')
                rela_type = annotation.split(' ')[0]
                try:
                    start_entity = entity[annotation.split(' ')[1].split(':')[1]]
                    end_entity = entity[annotation.split(' ')[2].split(':')[1]]
                except Exception:
                    print('start',annotation.split(' ')[1].split(':')[1])
                    print('end',annotation.split(' ')[2].split(':')[1])
                rela[id] = [rela_type]+ start_entity + end_entity
        f.close()
        entity_list = list(entity.values())
        entity_list.sort(key=operator.itemgetter(1))#對標注的實體位置的索引從小到大排序
        return entity_list, rela

def merge_ann(data_file, operation):
    rawtext = read_txt(data_file)
    if rawtext is None:
        return None
    entities, rela = read_ann(data_file)
    annote = list()
    last_end_index = 0
    startindex = 0
    gened_rela = list()
    interval_text=''
    for i in range(len(entities)):
        interval_text = rawtext[last_end_index:entities[i][1]]
        if last_end_index > 0 and ('。' in interval_text or '\n' in interval_text):  #至少第二个实体
            endindex = i    #
            gened_rela.extend(rela_generator(entities[startindex:endindex]))   #以一个完整句子为划分，组合这个句子中的实体对。
            startindex = endindex
        if len(interval_text)>1:
            words = " ".join(jieba.cut(interval_text, HMM=True))
            print(words)
            for word in words.split(" "):
                annote.append(word+'\tothers')
        elif len(interval_text)== 1:
            annote.append(interval_text + '\tothers')
        else:
            print(interval_text)
        #annote.extend((int(entities[i][1]) - last_end_index) * ['others'])
        #word_len = int(entities[i][2]) - int(entities[i][1])
        annote.append(entities[i][3]+'\t'+entities[i][0])
        last_end_index = int(entities[i][2])

    rela_extract(rela, gened_rela)
    interval_text = rawtext[last_end_index:len(interval_text)]
    if len(interval_text) > 1:
        words = " ".join(jieba.cut(interval_text, HMM=True))
        print(words)
        for word in words.split(" "):
            annote.append(word + '\tothers')
    elif len(interval_text) == 1:
        annote.append(interval_text + '\tothers')
    else:
        print(interval_text)

    text=''
    for i in range(len(annote)):
        text+=annote[i]+'\n'
    write_train_file(text, operation)
    return text

def rela_extract(rela, gened_relas):
    reall_relas = list(rela.values())
    count = 0
    for item in gened_relas:
        if item not in reall_relas:
            item[0]='others'
        else:
            count+=1
    relas = gened_relas
    relas.sort(key=operator.itemgetter(2))
    text = ""
    print(count)
    for i in range(len(relas)):
        text += relas[i][0]+'\t' #关系类型
        text += str(relas[i][1]) + '\t' + str(relas[i][4]) + '\t'+ str(relas[i][2]) + '\t'+ str(relas[i][3]) + '\t' #源实体1类型，源实体1单词
        text += str(relas[i][5]) + '\t' + str(relas[i][8]) + '\t'+ str(relas[i][6]) + '\t'+ str(relas[i][7]) + '\t'#源实体2类型，源实体2单词
        text += str(abs(int(relas[i][2]) - int(relas[i][6]))) + '\n'#两实体开始位置的相对间隔
    write_train_file(text, 'rela_train')
    return relas

def rela_generator(entities):
    relas = ['characteristic_of','image_feature_of','result_of','location_of','before','at','during','after','num_of','size_of','subject_of','negation_of','body_location_of']
    negation = ['negation']
    uncertainity = ['uncertainty']
    body_location = ['body_location']
    gened_rela = list()
    text = ""
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:  #排除自身配对
                continue
            entity_start, entity_end = entities[i], entities[j]
            if entity_start[0] in ['negation']:
                rela_type = 'negation_of'
            elif entity_start[0] in ['uncertainity']:
                rela_type = 'uncertainty_of'
            elif entity_start[0] in ['body_location'] and entity_end[0] in ['disorder','test','operation','negative_symptom','image_feature','pathology_feature']:
                rela_type = 'body_location_of'
            elif entity_start[0] in ['image_feature'] and entity_end[0] in ['test']:
                rela_type = 'image_feature_of'
            elif entity_start[0] in ['pathology_feature'] and entity_end[0] in ['test']:
                rela_type = 'pathology_feature_of'
            elif entity_start[0] in ['size'] and entity_end[0] in ['disorder','image_feature','pathology_feature']:
                rela_type = 'size_of'
            elif entity_start[0] in ['number'] and entity_end[0] in ['disorder']:
                rela_type = 'number_of'
            elif entity_start[0] in ['test_result','negative_symptom'] and entity_end[0] in ['test']:
                rela_type = 'result_of'
            elif entity_end[0] in ['period']:
                rela_type = 'during'
            elif entity_start[0] in ['onset_characteristic','condition_change'] and entity_end[0] in ['disorder']:
                rela_type = 'characteristic_of'
            elif entity_start[0] in ['test','operation','drug'] and entity_end[0] in ['hospital']:
                rela_type = 'location_of'
            else:
                continue

            gened_rela.append([rela_type]+entity_start+entity_end)
            text += rela_type + '\t'  # 关系类型
            text += str(entity_start[0]) + '\t' + str(entity_start[3]) + '\t'  # 源实体1类型，源实体1单词
            text += entity_end[0] + '\t' + entity_end[3] + '\t'  # 源实体2类型，源实体2单词
            text += str(abs(int(entity_end[1]) - int(entity_start[1]))) + '\n'  # 两实体开始位置的相对间隔
    return gened_rela
    #write_train_file(text, 'rela_train')



def write_train_file(text,operation):
    with open(filepath + operation+'.txt','a') as f:
        f.write(text)
        f.close()

if __name__ == '__main__':
    #if os.path.isdir(raw_data_file):
    folders = os.listdir(raw_data_file)
    for folder in folders:
        if '.' not in folder:
            print("The folder-%s is converting..." % folder)
            merge_ann(raw_data_file+'/'+folder+'/',args.operation)#train,dev,test
            print("Converting folder-%s is done!" % folder)
