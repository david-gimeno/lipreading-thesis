import os
import pandas as pd
from num2words import num2words
from tqdm import tqdm
import re

def clean_sentence(text):
    text = text.replace('{', '').replace('}', '')
    #text = re.sub(r"([A-Z]+)([2])([A-Z]+)", r"\1 TO \3", text)
    cleanText = ''
    for word in text.split(' '):
        original = word

        if any(map(str.isdigit, word)): #(not "'" in word) and (not word.isalpha()):
            word = re.sub(r"(THE)([0-9])", r"\1 \2", word).strip()
            if bool(re.match(r"([0-9]+)(ST|ND|RD|TH|THS)$", word)):
                word = word.replace('ST', '').replace('ND', '').replace('RD', '').replace('THS', ' S').replace('TH', '')
                to = 'ordinal'
            elif bool(re.match(r"[1|2][0-9]{3}(|'S)$", word)):
                if bool(re.match(r"[1|2][0-9]{3}'S$", word)):
                    word = word.split("'")[0] + " " + "\'S"
                to = 'year'
            elif bool(re.match(r"[0-9]{2}'S$", word)):
                word = word.split("'")[0] + " " + "\'S"
                to = 'cardinal'
            else:
                word = re.sub(r"([A-Z]+)([2])([A-Z]+)", r"\1 TO \3", word)
                word = re.sub(r"([0-9]+)M", r"\1 MILLIONS", word).strip()
                word = re.sub(r"([0-9]+)(PM)", r"\1 \2", word).strip()
                word = re.sub(r"([0-9]+)(AM)", r"\1 \2", word).strip()
                word = re.sub(r"([0-9]+)(D)", r"\1 \2", word).strip()
                word = re.sub(r"([A-Z]+)([0-9]+)", r"\1 \2", word).strip()
                word = re.sub(r"([0-9]+)([A-Z]+)", r"\1 \2", word).strip()
                word = re.sub(r"([0-9]+)(S$)", r"\1 \2", word).strip()
                word = " ".join(word.split("'"))
                to = 'cardinal'

            for subword in word.split(' '):
                if not any(map(str.isdigit, subword)):
                    cleanText += subword + ' '
                else:
                    #print(original, word)
                    cleanText += num2words(subword, to=to).replace('-', ' ').replace(',', ' ').upper() + ' '
        else:
            if '0' in word:
                print(word)
            cleanText += word + ' '

    cleanText = cleanText.strip()
    return cleanText
