# This file is written by Zhen Zhang, under the guideline of Prof. Jason Eisner in JHU's
# NLP course material. The dataset is available at "http://www.cs.jhu.edu/~jason/465/hw-hmm/data/".

import re
import collections
import numpy as np
from tools import *


class HMM:

    def __init__(self):
        # initialization
        self.tt_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.tw_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.wt_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.tt_singleton = collections.defaultdict(lambda: 0)
        self.tw_singleton = collections.defaultdict(lambda: 0)
        # fit result
        self.tag_list = collections.defaultdict(lambda: 0)
        self.tt_percent_dict = None
        self.tw_percent_dict = None
        # predict process
        self.optimal_dict = list()
        self.backpointer = list()
        self.real_tag = list()

    @staticmethod
    def parse(text):
        word, tag = re.findall("[^\n|^/]+", text)
        return word.lower(), tag

    def unsmooth_log(self, dic):
        for key, value in dic.items():
            total = sum(value.values())
            alpha = 1 + self.tt_singleton[key]
            for ke, val in value.items():
                backoff = val/total
                dic[key][ke] = (val + backoff * alpha) / (self.tag_list[key] + alpha)
        return dic

    def smooth_log(self, dic):
        word_num = len(self.wt_dict)
        result_dic = collections.defaultdict(lambda: 0)
        for key, value in dic.items():
            total = sum(value.values()) + word_num
            alpha = 1 + self.tw_singleton[key]
            result_dic[key] = collections.defaultdict(lambda: np.log(1 / total))
            for ke, val in value.items():
                backoff = np.log((val + 1) / total)
                result_dic[key][ke] = (val + backoff * alpha) / (self.tag_list[key] + alpha)
        return result_dic

    @staticmethod
    def rmse(real_tag, predict_tag):
        # mse
        count = 0
        for i in range(len(real_tag)):
            if real_tag[i] == predict_tag[i]:
                count += 1
        mse = count / len(real_tag)
        return mse

    def fit_each(self, word, tag0, tag1):
        self.tt_dict[tag0][tag1] += 1
        self.tw_dict[tag1][word] += 1
        self.wt_dict[word][tag1] += 1
        self.tag_list[tag1] += 1

        if self.tt_dict[tag0][tag1] == 1:
            self.tt_singleton[tag0] += 1
        elif self.tt_dict[tag0][tag1] == 2:
            self.tt_singleton[tag0] -= 1

        if self.tw_dict[tag1][word] == 1:
            self.tw_singleton[tag1] += 1
        elif self.tw_dict[tag1][word] == 2:
            self.tw_singleton[tag1] -= 1
        return self

    def fit(self, file):
        with open(file) as f:
            word, tag0 = self.parse(f.readline())
            self.tw_dict[tag0][word] += 1
            self.wt_dict[word][tag0] += 1
            for line in f:
                word, tag1 = self.parse(line)
                self.fit_each(word, tag0, tag1)
                tag0 = tag1
        self.tt_percent_dict = self.unsmooth_log(self.tt_dict)
        self.tw_percent_dict = self.unsmooth_log(self.tw_dict)
        return self

    def refresh(self):
        # renew predict data structures
        self.backpointer = [collections.defaultdict(lambda: 0)]
        self.backpointer[0]["###"] = "default"
        self.optimal_dict = [collections.defaultdict(lambda: -float('Inf'))]
        self.optimal_dict[0]["###"] = 0

    def backtrace(self, ptr):
        predict_tag = [None] * (ptr + 1)
        predict_tag[ptr] = "###"
        for i in range(ptr - 1, -1, -1):
            predict_tag[i] = self.backpointer[i + 1][predict_tag[i + 1]]
        return predict_tag

    def predict_each(self, word, ptr):
        self.backpointer.append(collections.defaultdict(lambda: 0))
        self.optimal_dict.append(collections.defaultdict(lambda: -float('Inf')))
        for key, _ in self.wt_dict[word].items():
            # key is current period tag
            for ke, val in self.optimal_dict[ptr - 1].items():
                # ke is previous period tag
                p = self.tw_percent_dict[key][word] + self.tt_percent_dict[ke][key]
                mu = p + val
                if mu > self.optimal_dict[ptr][key]:
                    self.backpointer[ptr][key] = ke
                    self.optimal_dict[ptr][key] = mu
        if self.backpointer[ptr].__len__() == 0:
            for key in self.tag_list.keys():
                # key is current period tag
                for ke, val in self.optimal_dict[ptr - 1].items():
                    # ke is previous period tag
                    p = self.tw_percent_dict[key][word] + self.tt_percent_dict[ke][key]
                    mu = p + val
                    if mu > self.optimal_dict[ptr][key]:
                        self.backpointer[ptr][key] = ke
                        self.optimal_dict[ptr][key] = mu
        return self

    def viterbi(self, file):
        self.refresh()
        real_tag = ["###"]
        ptr = 0

        with open(file) as f:
            f.readline()
            for line in f:
                ptr += 1
                word, tag = self.parse(line)
                real_tag.append(tag)
                self.predict_each(word, ptr)

        predict_tag = self.backtrace(ptr)
        # mse
        # mse = self.rmse(real_tag, predict_tag)
        return predict_tag

    def predict_sentence(self, string):
        self.refresh()
        ptr = 0

        sentence = string.split(" ")
        sentence.append("###")
        for word in sentence:
            ptr += 1
            self.predict_each(word, ptr)

        predict_tag = self.backtrace(ptr)
        return predict_tag

    def predict_sentence_simply(self, string_list):
        self.refresh()
        ptr = 0

        for word in string_list:
            ptr += 1
            self.predict_each(word, ptr)

        predict_tag = self.backtrace(ptr)
        return predict_tag

    def predict_type(self, test_word, pre, nex):
        sentence = "%s %s %s" % (pre, test_word, nex)
        predict_tag = self.predict_sentence(sentence)
        return predict_tag


if __name__ == "__main__":
    hmm = HMM()
    hmm.fit("entrain.txt")
    # print(hmm.tag_list)
    # print(open("test.txt").read())

    # file_sentences = parse_file("test_corpos.txt")
    # combined_sentences = list(map(parse_string, file_sentences))
    # wrong_sentences, correct_sentences = merge_sentences(combined_sentences)

    n = 3
    print(hmm.predict_sentence_simply(["###", "the", "campus", "is", "retiring", "the", "moobilenetx",
                                       "wireless", "network", "on", "June", ".", "###"]))
    # print(hmm.predict_sentence_simply(correct_sentences[n]))
    # print(wrong_sentences[n])
    # print(correct_sentences[n])

    # print(hmm.predict_type("years", "before", "the"))
    # print(hmm.viterbi("test.txt"))
