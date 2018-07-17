"""
Created on 2018-06-13
class: RL4SRD
@author: fengyue
"""

# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import json
import yaml
import copy
import math
import random
import numpy as np
import tensorflow as tf
import subprocess

# tf Graph input
input_query = tf.placeholder(tf.float32, [1, 100])
query_selected = tf.placeholder(tf.float32, [None, 100])
candidate = tf.placeholder(tf.float32, [None, 100])
action = tf.placeholder(tf.float32, [1, None])
G_t = tf.placeholder(tf.float32, [1, 1])


class RL4SRD(object):
    """docstring for RL4SRD"""
    def __init__(self, fileQueryPermutaion, fileQueryRepresentation, fileDocumentRepresentation, fileQueryDocumentSubtopics, folder):
        super(RL4SRD, self).__init__()

        with open(fileQueryPermutaion) as self.fileQueryPermutaion:
            self.dictQueryPermutaion = json.load(self.fileQueryPermutaion)

        with open(fileQueryRepresentation) as self.fileQueryRepresentation:
            self.dictQueryRepresentation = json.load(self.fileQueryRepresentation)
        for query in self.dictQueryRepresentation:
            self.dictQueryRepresentation[query] = np.matrix([self.dictQueryRepresentation[query]], dtype=np.float)
            self.dictQueryRepresentation[query] = np.transpose(self.dictQueryRepresentation[query])

        with open(fileDocumentRepresentation) as self.fileDocumentRepresentation:
            self.dictDocumentRepresentation = json.load(self.fileDocumentRepresentation)
        for doc in self.dictDocumentRepresentation:
            self.dictDocumentRepresentation[doc] = np.matrix([self.dictDocumentRepresentation[doc]], dtype=np.float)
            self.dictDocumentRepresentation[doc] = np.transpose(self.dictDocumentRepresentation[doc])

        with open(fileQueryDocumentSubtopics) as self.fileQueryDocumentSubtopics:
            self.dictQueryDocumentSubtopics = json.load(self.fileQueryDocumentSubtopics)

        self.folder = folder
        with open(self.folder + '/config.yml') as self.confFile:
            self.dictConf = yaml.load(self.confFile)
        self.learning_rate = self.dictConf['learning_rate']
        self.listTestSet = self.dictConf['test_set']
        self.lenTrainPermutation = self.dictConf['length_train_permutation']
        self.step = self.dictConf['step']
        self.gamma = 1.0
        self.hidden_dim = self.dictConf['hidden_dim']

        self.fileResult = open('result_' + sys.argv[1] + '.txt', 'w')
        

    def alphaDCG(self, alpha, query, docList, k):
        DCG = 0.0
        subtopics = []
        for i in xrange(20):
            subtopics.append(0)
        for i in xrange(k):
            G = 0.0
            if docList[i] not in self.dictQueryDocumentSubtopics[query]:
                continue
            listDocSubtopics = self.dictQueryDocumentSubtopics[query][docList[i]]
            if len(listDocSubtopics) == 0:
                    G = 0.0
            else:
                for subtopic in listDocSubtopics:
                    G += (1-alpha) ** subtopics[int(subtopic)-1]
                    subtopics[int(subtopic)-1] += 1
            DCG += G/math.log(i+2, 2)
        return DCG

    
    def sample_action(self, query, selected_list, selected_list_repr, doc_list):
        c = []
        c_id = []
        for can in doc_list:
            if can not in selected_list:
                doc_repr = carpe_diem.dictDocumentRepresentation[can]
                doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                c.append(doc_repr)
                c_id.append(can)

        if len(selected_list) == 0:
        	pred_prob = sess.run(doc_pred_first, feed_dict={input_query: [query], candidate: c})
        else:
            pred_prob = sess.run(doc_pred, feed_dict={input_query: [query], query_selected: selected_list_repr, candidate: c})


        tmp = random.random()
        sum_p = 0
        for id_p, p in enumerate(pred_prob):
            sum_p += p
            if tmp < sum_p:
                selected_list.append(c_id[id_p])
                break

        doc_repr = carpe_diem.dictDocumentRepresentation[c_id[id_p]]
        doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
        selected_list_repr.append(doc_repr)

        id_p_one_hot = np.zeros((1, len(c)))
        id_p_one_hot[0, id_p] = 1

        return selected_list, selected_list_repr, c, id_p_one_hot

    def take_action(self, query, selected_list, selected_list_repr, doc_list):
        c = []
        c_id = []
        for can in doc_list:
            if can not in selected_list:
                doc_repr = carpe_diem.dictDocumentRepresentation[can]
                doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                c.append(doc_repr)
                c_id.append(can)

        if len(selected_list) == 0:
        	pred_prob = sess.run(doc_pred_first, feed_dict={input_query: [query], candidate: c})
        else:
            pred_prob = sess.run(doc_pred, feed_dict={input_query: [query], query_selected: selected_list_repr, candidate: c})

        id_p = np.argmax(pred_prob)
        selected_list.append(c_id[id_p])

        doc_repr = carpe_diem.dictDocumentRepresentation[c_id[id_p]]
        doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
        selected_list_repr.append(doc_repr)

        return selected_list, selected_list_repr


def build_model(carpe_diem):
    V = tf.Variable(tf.random_uniform([100, carpe_diem.hidden_dim*2], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim))
    
    V_c = tf.Variable(tf.random_uniform([100, carpe_diem.hidden_dim], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim)) 
    V_h = tf.Variable(tf.random_uniform([100, carpe_diem.hidden_dim], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim))
    
    q_state_c = tf.sigmoid(tf.matmul(input_query, V_c))
    q_state_h = tf.sigmoid(tf.matmul(input_query, V_h))
    q_state = tf.concat([q_state_c, q_state_h], 1)
    
    # select first doc
    logits_first = tf.reshape(tf.matmul(tf.matmul(candidate, V), tf.transpose(q_state)), [-1])
    prob_first = tf.nn.softmax(logits_first)
    
    with tf.variable_scope('loss_first_policy'):
        loss_first = - tf.matmul(G_t, tf.matmul(action, tf.reshape(tf.log(tf.clip_by_value(prob_first, 1e-30, 1.0)), [-1, 1])))
        optimizer_first = tf.train.AdagradOptimizer(carpe_diem.learning_rate).minimize(loss_first)

    input = tf.reshape(query_selected, [1, -1, 100])
    with tf.variable_scope("LSTM") as vs:
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=carpe_diem.hidden_dim, state_is_tuple=False)
    _, states = tf.nn.dynamic_rnn(rnn_cell, input, initial_state=q_state, dtype=tf.float32)  # [1, dim]
    logits = tf.reshape(tf.matmul(tf.matmul(candidate, V), tf.transpose(states)), [-1])
    prob = tf.nn.softmax(logits)
    
    with tf.variable_scope('loss_policy'):
        loss = - tf.matmul(G_t, tf.matmul(action, tf.reshape(tf.log(tf.clip_by_value(prob, 1e-30, 1.0)), [-1, 1])))
        optimizer = tf.train.AdagradOptimizer(carpe_diem.learning_rate).minimize(loss)
        
    return optimizer, prob, optimizer_first, prob_first


query_permutation_file = '../data/query_permutation.json'
query_representation_file = '../data/query_representation.dat'
document_representation_file = '../data/doc_representation.dat'
query_document_subtopics_file = '../data/query_doc.json'
folder = '../data/' + sys.argv[1]


carpe_diem = RL4SRD(query_permutation_file, query_representation_file, document_representation_file, query_document_subtopics_file, folder)
opt, doc_pred, opt_first, doc_pred_first = build_model(carpe_diem)

saver = tf.train.Saver(max_to_keep=0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ckpt = tf.train.get_checkpoint_state('model_' + sys.argv[1] + '/')
if ckpt and ckpt.model_checkpoint_path:
    print 'Load model from:', ckpt.model_checkpoint_path
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

listKeys = carpe_diem.dictQueryPermutaion.keys()
iteration = 0

while True:
    for query_id in listKeys:
        if int(query_id) in carpe_diem.listTestSet:
            continue
        
        q = carpe_diem.dictQueryRepresentation[query_id]
        q = np.reshape(np.asarray(q), -1).tolist()

        listPermutation = copy.deepcopy(carpe_diem.dictQueryPermutaion[query_id]['permutation'])
        idealScore = carpe_diem.alphaDCG(0.5, query_id, listPermutation, carpe_diem.lenTrainPermutation)
        if idealScore == 0:
            continue

        #sample
        listSelectedSet = []
        listSelectedSet_repr = []
        candidate_set = []
        candidate_set_id = []

        while len(listSelectedSet) < carpe_diem.lenTrainPermutation:
            listSelectedSet, listSelectedSet_repr, one_candidate_set, one_candidate_set_id = carpe_diem.sample_action(q, listSelectedSet, listSelectedSet_repr, listPermutation)
            candidate_set.append(one_candidate_set)
            candidate_set_id.append(one_candidate_set_id)

        score = carpe_diem.alphaDCG(0.5, query_id, listSelectedSet, carpe_diem.lenTrainPermutation)

        #optimizer
        for id_i in range(carpe_diem.lenTrainPermutation):
            g_t = score - carpe_diem.alphaDCG(0.5, query_id, listSelectedSet, id_i)
            if id_i == 0:
                sess.run(opt_first, feed_dict={input_query: [q], candidate: candidate_set[id_i], action: candidate_set_id[id_i], G_t: [[g_t]]})
            else:
                sess.run(opt, feed_dict={input_query: [q], query_selected: listSelectedSet_repr[:id_i], candidate: candidate_set[id_i], action: candidate_set_id[id_i], G_t: [[g_t]]})
            
    ## test
    if iteration % 50 == 0:
        floatSumResultScore_ndcg_5 = 0.0
        floatSumResultScore_ndcg_10 = 0.0
        
        resultCount = 0.0

        fileTmpResult_policy = open('tmp_result_' + sys.argv[1] + '.txt', 'w')
        
        for query_test in carpe_diem.listTestSet:
            listSelectedSet = []
            listSelectedSet_repr = []
            listTest = copy.deepcopy(carpe_diem.dictQueryPermutaion[str(query_test)]['permutation'])
            idealScore_ndcg_10 = carpe_diem.alphaDCG(0.5, str(query_test), listTest, 10)
            idealScore_ndcg_5 = carpe_diem.alphaDCG(0.5, str(query_test), listTest, 5)
            if idealScore_ndcg_5 == 0 or idealScore_ndcg_10 == 0:
                continue
            random.shuffle(listTest)
            q_test = carpe_diem.dictQueryRepresentation[str(query_test)]
            q_test = np.reshape(np.asarray(q_test), -1).tolist()
            
            while len(listSelectedSet) < carpe_diem.lenTrainPermutation:
                listSelectedSet, listSelectedSet_repr = carpe_diem.take_action(q_test, listSelectedSet, listSelectedSet_repr, listTest)
            
            score = carpe_diem.alphaDCG(0.5, str(query_test), listSelectedSet, carpe_diem.lenTrainPermutation)          

            # save result
            for id_num, doc_id_selected in enumerate(listSelectedSet):
                fileTmpResult_policy.write(str(query_test) + ' Q0 ' + doc_id_selected + ' ' +str(id_num+1) + ' ' + str(len(listSelectedSet)-id_num) + ' ' + sys.argv[1] + '\n')
                fileTmpResult_policy.flush()

            resultScore_ndcg_10 = carpe_diem.alphaDCG(0.5, str(query_test), listSelectedSet, 10)
            resultScore_ndcg_5 = carpe_diem.alphaDCG(0.5, str(query_test), listSelectedSet, 5)
            floatSumResultScore_ndcg_5 += resultScore_ndcg_5 / idealScore_ndcg_5
            floatSumResultScore_ndcg_10 += resultScore_ndcg_10 / idealScore_ndcg_10
            
            resultCount += 1
            
        result_ndcg_5 = floatSumResultScore_ndcg_5 / resultCount
        result_ndcg_10 = floatSumResultScore_ndcg_10 / resultCount
        
        # metrics
        p_can = subprocess.Popen(['./ndeval', '../metrics/my_qrels.txt', 'tmp_result_' + sys.argv[1] + '.txt'], shell=False, stdout=subprocess.PIPE, bufsize=-1)
        output_eval = p_can.communicate()
        output_eval = output_eval[-2].split('\n')[-2]
        output_eval = output_eval.split(',')
        metrics_err_5 = output_eval[2]
        metrics_err_10 = output_eval[3]
        metrics_ndcg_5 = output_eval[11]
        metrics_ndcg_10 = output_eval[12]
        metrics_srecall_5 = output_eval[20]
        metrics_srecall_10 = output_eval[21]

        carpe_diem.fileResult.write(str(iteration) + ' ' + str(result_ndcg_5) + ' ' + str(result_ndcg_10) + '\n')
        carpe_diem.fileResult.write(str(iteration) + ' ' + metrics_ndcg_5 + ' ' + metrics_ndcg_10 + ' ' + metrics_srecall_5 + ' ' + metrics_srecall_10 + ' ' + metrics_err_5 + ' ' + metrics_err_10 + '\n')
        carpe_diem.fileResult.write('\n')
        carpe_diem.fileResult.flush()

        saver.save(sess, 'model_' + sys.argv[1] + '/' + 'model.ckpt', global_step=iteration)
        print 'Save model @ EPOCH %d' % iteration

    iteration += 1

print "Game over!"
