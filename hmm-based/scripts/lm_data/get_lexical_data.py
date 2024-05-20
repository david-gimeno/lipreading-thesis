# -*- coding: utf-8 -*-

import os
import sys
from tqdm import tqdm
from g2p_en import G2p
sys.path.insert(0,'./scripts/')
from my_utils import *

def get_vocabulary(lm_data):
    vocabulary = {}
    for text_sample in lm_data:
        for i, word in enumerate(text_sample.strip().split(" ")):
            if vocabulary.get(word) == None:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1
    print("\nOur vocabulary is made up of " + str(len(vocabulary.keys())) + " different words")
    return vocabulary

def get_lexicon_txt(vocabulary, phonemizer):
    ## LEXICON.TXT FILE ##
    with open(dst_path + "/lexicon.txt", "w", encoding="utf-8") as f:
        ## ADDING SPECIAL KALDI SILENCE SYMBOLS ##
        f.write("!SIL sil\n<UNK> spn\n")

        phones = {}
        for word in tqdm(sorted(vocabulary.keys())):
            if len(word) > 0:
                if "character" in lm_level:
                    phonemas = word
                elif "word" in lm_level:
                    if phonemizer is not None:
                        phonemas = collapse_english_phonemes(phonemizer(word))
                    else:
                        command = "echo " + str(word) + " | ./scripts/eutranscribe"
                        phonemas = " ".join(os.popen(command).read()).strip()

                for phone in phonemas.split():
                    if phones.get(phone) == None:
                        phones[phone] = 1
                    else:
                        phones[phone] += 1

                f.write(word + " " + phonemas + "\n")
    # print(phones)
    print("Our phonetic system is made up of " + str(len(phones.keys())) + " different phones\n")
    return phones

def get_optional_silence_phones_txt():
    ## OPTIONAL_SILENCE.TXT  ##
    with open(dst_path + "/optional_silence.txt", "w", encoding="utf-8") as f:
        f.write("sil\n")

def get_silence_phones_txt():
    ## SILENCE_PHONES.TXT ##
    with open(dst_path + "/silence_phones.txt", "w", encoding="utf-8") as f:
        f.write("sil\nspn\n")

def get_nonsilence_phones_txt(phones):
    ## NONSILENCE_PHONES.TXT ##
    with open(dst_path + "/nonsilence_phones.txt", "w", encoding="utf-8") as f:
        for phone in sorted(phones.keys()):
            f.write(phone + "\n")

if __name__ == "__main__":
    text_path = sys.argv[1]
    dst_path = sys.argv[2]
    lm_level = sys.argv[3]

    database = dst_path.split("/")[2]
    os.makedirs(dst_path, exist_ok=True)

    if database in ["LRS2-BBC", "LRS3-TED"]:
        phonemizer = G2p()
    elif database in ["VLRF", "CMU-MOSEAS-Spanish", "LIP-RTVE", "Multilingual-TEDx-Spanish"]:
        phonemizer = None

    enc = "ISO-8859-1" if "VLRF" in dst_path else "utf-8"
    lm_data = [l.strip() for l in open(text_path, "r", encoding=enc).readlines()]

    vocabulary = get_vocabulary(lm_data)
    phones = get_lexicon_txt(vocabulary, phonemizer)
    get_optional_silence_phones_txt()
    get_silence_phones_txt()
    get_nonsilence_phones_txt(phones)
