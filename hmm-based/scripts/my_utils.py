import os
from unidecode import unidecode
from num2words import num2words

def clean_spanish_text(sentence):
    cleanSentence = sentence.lower()
    cleanSentence = cleanSentence.replace("ñ","N")
    cleanSentence = unidecode(cleanSentence.strip())

    return cleanSentence

def clean_english_text(sentence):
    return sentence.replace(",", "").replace("{", "").replace("}", "")

def collapse_english_phonemes(vocab_phonemes):
    # " ".join([p for p in g2p(vocab) if p not in ["", " "]]).strip()
    result = []
    for p in vocab_phonemes:
        if p not in [",", "'", "", " "]:
            if ("0" in p) or ("1" in p) or ("2" in p):
                result.append(p[:-1])
            else:
                result.append(p)

    return " ".join(result)
