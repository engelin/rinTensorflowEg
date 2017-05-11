#-*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys


# matplot for Korean
reload(sys)
sys.setdefaultencoding('utf-8')

"""
sentences = ["나 고양이 좋다", "나 강아지 좋다", "나 동물 좋다",
             "강아지 고양이 동물", "애인 고양이 강아지 좋다", 
             "고양이 생선 우유 좋다", "강아지 고양이 눈 좋다",
             "나 애인 좋다", "애인 나 싫다",
             "애인 나 영화 책 음악 좋다", "나 게임 만화 애니 싫다",
             "고양이 강아지 싫다", "강아지 고양이 좋다"]
"""

sentences = ["I like cat", "I like dog", "I like animal",
            "dog cat animal", "you like cat dog",
            "cat like fish milk", "dog like cat snow",
            "I like you", "you hate I", "you I like movie book music",
            "I hate game animation", "cat hate dog", "dog like cat"]


# spilt the sentences and set list by the eigen words
word_list = " ".join(sentences).split()
word_list = list(set(word_list))

# word to num
word_dict = {w: i for i, w in enumerate(word_list)}
word_index = [word_dict[word] for word in word_list]


#######################################################################
# set skip-gram model with 1-size window                              #
# e.g) 나 게임 만화 애니 싫다                                         #
#     1step: ([나 만화] 게임), ([게임 애니] 만화), ([만화 싫다] 애니) #
#     2step: (게임 나), (게임 만화), (만화 게임),                     #
#            (만화 애니), (애니 만화), (애니 싫다)                    #
#######################################################################
skip_grams = []

for i in range(1, len(word_index) - 1):
   # (context, target): ([target index - 1, target index + 1], target)
   target = word_index[i]
   context = [word_index[i - 1], word_index[i + 1]]

   # (target, context[0]), (target, context[1])...
   for w in context:
      skip_grams.append([target, w])


def random_batch(data, size):
   random_inputs = []
   random_labels = []
   random_index = np.random.choice(range(len(data)), size, replace=False)

   for i in random_index:
      random_inputs.append(data[i][0])   # target
      random_labels.append([data[i][1]]) # context word

   return random_inputs, random_labels


# variables setting
training_epoch = 300
learning_rate = 0.1
batch_size = 20
# dimension of word vectors: 2
embedding_size = 2
# sampling size of nce_loss function for learning the word2vec model
# smaller than batch_size
num_sampled = 15
# # of total word
voc_size = len(word_list)


inputs = tf.placeholder(tf.int32, shape=[batch_size])
# To use tf.nn.nce_loss, output value consists of this.
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

#print "document", tf.nn.nce_loss.__doc__
#loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, selected_embed, labels, num_sampled, voc_size))
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
   init = tf.global_variables_initializer()
   sess.run(init)

   for step in range(1, training_epoch + 1):
      batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

      _, loss_val = sess.run([train_op, loss],
                             feed_dict={inputs: batch_inputs, labels: batch_labels})

      if step % 10 == 0:
         print "loss at step ", step, ": ", loss_val

   trained_embeddings = embeddings.eval()


for i, label in enumerate(word_list):
   x, y = trained_embeddings[i]
   plt.scatter(x, y)
   plt.annotate(label, xy=(x, y), xytext=(5, 2),
                textcoords='offset points', ha='right', va='bottom')

plt.show()
