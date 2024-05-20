def get_lrs3_test_vocab():
    dict = {}
    text = open("./data/LRS3-TED/100fps/speaker-independent/audio/test/text", "r").readlines()
    for line in text:
        words = line.strip().split()[1:]
        for word in words:
            if dict.get(word) == None:
                dict[word] = 1
            else:
                dict[word] += 1

    return dict

def get_lrs3_clean_dict():
    clean_list = open("./data/LRS3-TED/withnumber_dict.txt").readlines()
    clean_dict = {}
    for sample in clean_list:
        word = sample.split()[0]
        cleanword = " ".join(sample.split()[1:]).strip()

        if clean_dict.get(word) == None:
            clean_dict[word] = cleanword

    return clean_dict


def remove_num_id_word():
    words_num = open("./words.txt", "r").readlines()

    for word_num in words_num:
        word = word_num.strip().split()[0]
        print(word)

def remove_repeated_words_cmu():
    words = open("./data/LRS3-TED/lrs3_cmu_lexicon.txt", "r").readlines()

    for word in words:
        if ("(2)" not in word) and ("(3)" not in word) and ("(4)" not in word):
            line = " ".join(word.split())
            print(line)

def get_lrs3_lexicon_dict():
    lexicon_list = open("./data/LRS3-TED/lrs3_cmu_lexicon.txt").readlines()
    lexicon_dict = {}
    for sample in lexicon_list:
        word = sample.split()[0]
        phones = " ".join(sample.split()[1:]).strip()

        if lexicon_dict.get(word) == None:
            lexicon_dict[word] = phones

    return lexicon_dict

def get_number_dict():
    lexicon = open("./data/LRS3-TED/lrs3_cmu_lexicon.txt", "r").readlines()

    for wordphones in lexicon:
        word = wordphones.strip().split()[0]
        if True in [char.isdigit() for char in word]:
            print(word)

if __name__ == "__main__":
    remove_repeated_words_cmu()
    #remove_num_id_word()
    #print(get_lrs3_lexicon_dict())
    #get_number_dict()
    #print(get_lrs3_test_vocab().keys())
    #print(get_lrs3_clean_dict())
