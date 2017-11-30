from sklearn import svm as svm
from sklearn import model_selection
from sklearn.externals import joblib
import tensorflow as tf
import time
import helper
import sys, os
import numpy as np


dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir)
train_path = os.path.join(dir,'rawdatarela_train.txt')
embedding_path = os.path.join(dir,'embedding.npy')
save_path = os.path.join(dir,'model')
start_time = time.time()

word2id, id2word = helper.loadMap("word2id.txt")
label2id, id2label = helper.loadMap("label2id.txt")
entitylabel2id, id2entitylabel = helper.loadMap("entitylabel2id.txt")
num_words = len(id2word.keys())
num_classes = len(id2label.keys())
emb_dim=128
batch_size = 128
print("preparing train and validation data")
label, entity1label, entity2label, entity1, entity2, distance  = helper.getTrainData(train_path=train_path)
if embedding_path !=None:
    embedding_matrix = helper.getEmbedding(embedding_path, emb_dim=emb_dim)
else:
    embedding_matrix = None
features = np.transpose(np.array([entity1label, entity2label, distance]))
input_entity1_emb = np.zeros((len(entity1), emb_dim))
input_entity2_emb = np.zeros((len(entity2), emb_dim))
for i in range(len(entity1)):
    try:
        input_entity1_emb[i] = embedding_matrix[entity1[i]]
        input_entity2_emb[i] = embedding_matrix[entity2[i]]
    except Exception:
        continue
inputs = np.concatenate([input_entity1_emb, input_entity2_emb, features], 1)
X= inputs
y = np.array(label)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,shuffle=True,train_size=0.75)
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr', class_weight='balanced')#
clf.fit(X_train, y_train.ravel())
joblib.dump(clf,'svm_rela_extractor.m')
print(clf.score(X_train, y_train)) # 精度
y_hat = clf.predict(X_train)


def show_accuracy(y_hat, y_train, param):
    count = 0
    for p, t in zip(y_hat, y_train):
        if p == t:
            count += 1
    print(param + "准确率是%f" % (count / len(y_hat)))


show_accuracy(y_hat, y_train, '训练集')
print(clf.score(X_test, y_test))
y_hat = clf.predict(X_test)
show_accuracy(y_hat, y_test, '测试集')
print('决策函数:\n', clf.decision_function(X_train))
with open('decision_function.txt', 'a') as f:
    for line in clf.decision_function(X_train):
        f.write(str(line)+'\n')
    f.close()

result=clf.predict(X_test)
text =''
for rawdata,pred in zip(y_test,result):
    text += '真实标签：'+ id2label[rawdata] + '\t预测标签：' + id2label[pred] + '\n'
with open('result.txt','w') as f:
    f.write(text)
    f.close()
print('\n预测结果:\n', result)
run_time = (time.time() - start_time)/3600
print("Total running time: %f"%run_time)


