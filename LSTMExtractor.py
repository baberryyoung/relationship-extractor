import numpy as np
import tensorflow as tf
import math
import helper

class LSTMExtractor(object):
    def __init__(self, num_epochs=100, embedding_matrix=None, num_classes=1, num_words = None, is_training = True):
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 128
        self.num_layers = 1
        self.emb_dim = 128
        self.hidden_dim = 259
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.num_words = num_words
        self.num_steps = 1

        #placeholders of
        #self.inputs = tf.placeholder(tf.int32,[None])
        self.input_entity1 = tf.placeholder(tf.int32,[self.batch_size, 1])
        self.input_entity2 = tf.placeholder(tf.int32,[self.batch_size, 1])
        self.features = tf.placeholder(tf.int32, [self.batch_size, 3])
        self.targets = tf.placeholder(tf.float32, [self.batch_size, self.num_classes])

        if embedding_matrix.any() != None:
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_words, self.emb_dim])
        try:
            self.input_entity1_emb = tf.nn.embedding_lookup(self.embedding, self.input_entity1)
            self.input_entity2_emb = tf.nn.embedding_lookup(self.embedding, self.input_entity2)
        except Exception:
            pass
        self.input_entity1_emb = tf.reshape(self.input_entity1_emb,[self.batch_size,self.emb_dim])
        self.input_entity2_emb = tf.reshape(self.input_entity2_emb, [self.batch_size,self.emb_dim])
        self.features = tf.reshape(self.features, [self.batch_size, 3])
        self.inputs = tf.transpose(tf.concat([self.input_entity1_emb, self.input_entity2_emb, tf.cast(self.features, tf.float32)],1))

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.batch_size, self.hidden_dim])),
            'encoder_h2': tf.Variable(tf.random_normal([self.hidden_dim, self.emb_dim])),
            'decoder_h1': tf.Variable(tf.random_normal([self.emb_dim, self.hidden_dim])),
            'decoder_h2': tf.Variable(tf.random_normal([self.hidden_dim, self.batch_size])),

        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.hidden_dim])),
            'encoder_b2': tf.Variable(tf.random_normal([self.emb_dim])),
            'decoder_b1': tf.Variable(tf.random_normal([self.hidden_dim])),
            'decoder_b2': tf.Variable(tf.random_normal([self.batch_size])),
        }

        self.outputs = tf.transpose(self.decoder(self.encoder(self.inputs)))
        self.w = tf.get_variable("w", [self.hidden_dim, self.num_classes])
        self.b = tf.get_variable("b", [self.num_classes])
        #self.logits = tf.nn.softmax(tf.matmul(self.outputs, self.w) + self.b)

        self.logits = tf.matmul(self.outputs, self.w) + self.b

        self.cost = tf.nn.l2_loss(self.w) + tf.reduce_sum(tf.maximum(1 - self.targets * self.logits, 0))
        #train_step = tf.train.AdamOptimizer(0.01).minimize(self.cost)
        #self.y_predict = tf.sign(self.logits)
        self.y_predict = tf.argmax(self.logits, 1)
        #self.loss = tf.reduce_sum(tf.square(self.logits - self.targets))
        self.train_summary = tf.summary.scalar("loss", self.cost)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
        return layer_2

    # decoder,从128恢复到259个节点
    def decoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
        return layer_2


    def train(self, sess, X_train, y_train, X_val=None, y_val=None):
        saver = tf.train.Saver()

        num_iterations = int(math.ceil(1.0 * len(X_train)/self.batch_size))
        for epoch in range(self.num_epochs):# shuffle train in each epoch
            shuffle_index = np.arange(len(X_train))
            np.random.shuffle(shuffle_index)
            X_train = X_train[shuffle_index]
            y_train = y_train[shuffle_index]

            for iteration in range(num_iterations):
                X_train_batch, y_train_batch = helper.nextBatch(X_train, y_train, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                X_train_entity1, X_train_entity2, X_train_features = np.transpose(np.array([X_train_batch[:,0]])), X_train_batch[:,0:1], X_train_batch[:,1:-1]
                #y_train_batch = np.transpose(np.array([y_train_batch]))#列表需要转换成矩阵类型运算
                _, logits , loss_train,train_summary = sess.run([self.optimizer, self.y_predict, self.cost,  self.train_summary], feed_dict = {self.input_entity1: X_train_entity1, self.input_entity2: X_train_entity2, self.features: X_train_features, self.targets: y_train_batch})
                predict_train = logits
                if iteration!= 0 and iteration % 100 == 0:
                    precision_train, recall_train, f1_train = self.evaluate(X_train_batch, y_train_batch, predict_train)
                    print('Iteration:%d\tprecision:%f\trecall:%f\tf1:%f\n'%(iteration,precision_train,recall_train,f1_train))


    def evaluate(self, X, y, predicts):
        precision = - 1.0
        recall = -1.0
        hit_num = 0
        pred_num = 0
        true_num = 0
        y = helper.todigit(y)
        for i in range(len(y)):
            if y[i] == predicts[i]:
                hit_num += 1
                pred_num += len(predicts)
                true_num = len(y)

        if pred_num != 0:
            precision = hit_num / pred_num
        if true_num != 0:
            recall = hit_num / true_num
        f1 = 2 * precision * recall/(precision + recall)
        return precision, recall, f1

