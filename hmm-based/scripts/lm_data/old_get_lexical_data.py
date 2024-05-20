import os
import sys
from tqdm import tqdm
from g2p_en import G2p
sys.path.insert(0,'./scripts/')
from my_utils import *


g2p = G2p()

text_path = sys.argv[1]
dst_path = sys.argv[2]
lm_level = sys.argv[3]

database = dst_path.split("/")[2]
os.makedirs(dst_path, exist_ok=True)

enc = "ISO-8859-1" if "VLRF" in dst_path else "utf-8"
lm_data = [l.strip() for l in open(text_path, "r", encoding=enc).readlines()]

## GETTING VOCABULARY ##
vocabulary = {}
for text_sample in lm_data:
    for i, word in enumerate(text_sample.split(" ")):
        #if word.isdigit():
        #    word = num2words(word).upper()
        #    for w in word.split():
        #        if vocabulary.get(w) == None:
        #            vocabulary[w] = 1
        #        else:
        #            vocabulary[w] += 1
        #else:
        if vocabulary.get(word) == None:
            vocabulary[word] = 1
        else:
            vocabulary[word] += 1
print("\nOur vocabulary is made up of " + str(len(vocabulary.keys())) + " different words")

## LEXICON.TXT FILE ##
with open(dst_path + "/lexicon.txt", "w", encoding="utf-8") as f:
    ## ADDING SPECIAL KALDI SILENCE SYMBOLS ##
    f.write("!SIL sil\n<UNK> spn\n")
    phones = {}
    for vocab in tqdm(sorted(vocabulary.keys())):
        if "character" in lm_level:
            phonemas = vocab
        elif "word" in lm_level:
            if database in ["LRS2-BBC", "LRS3-TED"]:
                phonemas = collapse_english_phonemes(g2p(vocab))
                # phonemas = " ".join([p for p in g2p(vocab) if p not in ["", " "]]).strip()
            #if database in ["LIP-RTVE", "VLRF"]:
            #    if vocab.isalpha():
            #        command = "echo " + str(vocab) + " | ./local/eutranscribe"
            #        phonemas = " ".join(os.popen(command).read()).strip()

        for phone in phonemas.split(" "):
            if phone != " " and len(phone) > 0:
                if phones.get(phone) == None:
                    phones[phone] = 1
                else:
                    phones[phone] += 1

        f.write(vocab + " " + phonemas + "\n")
print("Our phonetic system is made up of " + str(len(phones.keys())) + " different phones\n")

## OPTIONAL_SILENCE.TXT  ##
with open(dst_path + "/optional_silence.txt", "w", encoding="utf-8") as f:
    f.write("sil\n")

## SILENCE_PHONES.TXT ##
with open(dst_path + "/silence_phones.txt", "w", encoding="utf-8") as f:
    f.write("sil\nspn\n")

## NONSILENCE_PHONES.TXT ##
with open(dst_path + "/nonsilence_phones.txt", "w", encoding="utf-8") as f:
    for phone in sorted(phones.keys()):
        f.write(phone + "\n")
