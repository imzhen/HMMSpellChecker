import re
import collections
import hmm
from tools import *


class Checker:

    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.model = collections.defaultdict(lambda: 1)
        with open('big.txt') as f:
            self.train(self.to_lower(f.read()))
        self.hmm = hmm.HMM()
        self.hmm.fit("entrain.txt")

    @staticmethod
    def to_lower(text):
        return re.findall('[a-z]+', text.lower())

    def train(self, words):
        for f in words:
            self.model[f] += 1

    def edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts = [a + c + b for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.model)

    def known(self, words):
        return set(w for w in words if w in self.model)

    def find_error(self, sentence):
        pos = 1
        while sentence[pos] != '###':
            word = sentence[pos]
            candidates = [self.known([word]), self.known(self.edits1(word)), self.known_edits2(word)]
            if candidates[0] == set():
                candidates = set.union(candidates[1], candidates[2])
                return pos, candidates
            pos += 1

    def correct_by_tag(self, sentence, pos, candidates):
        hmm_justified_tags = []
        candidates = list(candidates)
        for item in candidates:
            sentence[pos] = item
            item_tag = self.hmm.predict_sentence_simply(sentence)[2:pos + 3]
            hmm_justified_tags.append(tuple(item_tag))
        hmm_justified_tag = max(set(hmm_justified_tags), key=hmm_justified_tags.count)
        hmm_justified_candidates = []
        for i in range(len(candidates)):
            if hmm_justified_tags[i] == hmm_justified_tag:
                hmm_justified_candidates.append(candidates[i])
        return hmm_justified_candidates

    def correct_by_frequency(self, candidates):
        return max(candidates, key=self.model.get)

    def corrector(self, pair):
        word, sentence = pair[0], pair[1]
        sentence = ["###"] + re.findall(r"[\w']+|[.,!?;]", sentence) + ["###"]

        pos, candidates = self.find_error(sentence)
        candidates = self.correct_by_tag(sentence, pos, candidates)
        final_word = self.correct_by_frequency(candidates)

        # sentence[pos] = final_word
        return final_word


if __name__ == "__main__":
    file_sentences = parse_file("test_corpus.txt")

    checker = Checker()


    def parse_file(file_name):
        words = []
        sentences = []
        with open(file_name, encoding="utf-8") as f:
            for line in f:
                if line[0] == '/':
                    continue
                word, sentence = line.split("###")
                words.append(word)
                sentences.append(sentence)
        return words, sentences

    # test cases
    # print(checker.corrector(['###', "i", "haves", "told", "you", "to", "work", "hard", '.', '###']))
    words, sentences = parse_file("myfile_final.txt")
    pairs = list(zip(words, sentences))
    print(list(map(checker.corrector, pairs)))
    # print(checker.corrector('i need to check the water in my radiater before we leave .'))
    # print(checker.corrector(['###', 'i', 'felt', 'very', 'strange', 'at', 'break', 'time', 'when', 'the', 'brack',
    #                          'was', 'finished', 'in', 'the', 'winter', 'when', 'it', 'was', 'snowing', 'I', 'thought',
    #                          'it', 'was', 'a', 'ghost', 'everything', 'except', 'the', 'houses', '###']))

    # # Next are test data (test_corpus usage)
    # combined_sentences = list(map(parse_string, file_sentences))
    # wrong_sentences, correct_sentences = merge_sentences(combined_sentences)
    # corrected_sentences = list(map(checker.corrector, wrong_sentences))
    # sentences_pair = list(zip(corrected_sentences, correct_sentences))
    # accuracy = 0
    # for i in range(len(sentences_pair)):
    #     sentence_pair = sentences_pair[i]
    #     if " ".join(sentence_pair[0]) == " ".join(sentence_pair[1]):
    #         accuracy += 1
    # print(accuracy / len(sentences_pair))

