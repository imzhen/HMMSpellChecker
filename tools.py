from functools import reduce, partial
import re


def to_lower(text):
    return re.findall('[a-z]+', text.lower())


def to_lower_sentence(sentence_list):
    return ['###'] + reduce(lambda x, y: x+y, [to_lower(word) for word in sentence_list]) + ['.', '###']


def parse_file(file_path):
    """
    It is used to parse a file like test_corpos. Typically it should
    behave like this format:
    -----------------------
    $id
    words
    ...
    words
    $id
    -----------------------
    Then extract the words part into a list called sentence_list as a sublist,
    and for each sentence, there are words with typos and without typos. The
    next task is passed to parse_string function
    """
    sentence_list = []
    pos = -1
    with open(file_path) as f:
        for line in f:
            if line[0] == '$':
                pos += 1
                sentence_list.append([])
            else:
                # remove \n by subsetting -1
                sentence_list[pos].append(line[:-1])
    return sentence_list


def parse_string(sentences):
    """
    This function is served as changing each sublist into several sentences,
    based on the words. Each line must have one and only one typo, then for
    n lines, we will have n sentences, with each sentence have one typo.
    I will also build a corresponding list indicating which word is wrong,
    and the correct one.
    """
    words_list = list(map(lambda sentence: sentence.split(" "), sentences))

    def substitute(words, correct=True):
        """
        This function is a small inside function just convert each part into
        wrong one and right one
        """
        words_new = words.copy()
        pos = words_new.index("*")
        if correct:
            words_new[pos] = words_new[2]
            return words_new[4:]
        else:
            words_new[pos] = words_new[0]
            return words_new[4:]

    correct_each = list(map(partial(substitute, correct=True), words_list))
    wrong_each = list(map(partial(substitute, correct=False), words_list))

    correct_sentence = list(reduce(lambda a, b: a+b, correct_each))
    correct_sentence = correct_sentence
    correct_sentence = [correct_sentence] * len(correct_each)

    wrong_sentence = []
    for i in range(len(correct_each)):
        new_each = correct_each.copy()
        new_each[i] = wrong_each[i]
        new_sentence = list(reduce(lambda a, b: a+b, new_each))
        new_sentence = new_sentence
        wrong_sentence.append(new_sentence)

    return wrong_sentence, correct_sentence


def merge_sentences(combined_sentences):
    """
    After the previous step of finding each wrong and correct sentence
    in each id combination, I need to collect them all together and group
    into a new wrong and right sentence for further use
    """
    wrong_sentence = list(reduce(lambda x, y: x+y, [a[0] for a in combined_sentences]))
    wrong_sentence = list(map(to_lower_sentence, wrong_sentence))
    correct_sentence = list(reduce(lambda x, y: x+y, [a[1] for a in combined_sentences]))
    correct_sentence = list(map(to_lower_sentence, correct_sentence))
    return wrong_sentence, correct_sentence

if __name__ == "__main__":
    file_sentences = parse_file("test_corpos.txt")
    combined_sentences = list(map(parse_string, file_sentences))
    wrong_sentences, correct_sentences = merge_sentences(combined_sentences)
    print(wrong_sentences)
