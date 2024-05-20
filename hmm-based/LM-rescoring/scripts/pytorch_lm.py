import os
import copy
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from src.lipreading.asr.asr_utils import get_model_conf
from src.lipreading.nets.lm_interface import dynamic_import_lm
from src.lipreading.asr.asr_utils import torch_load

def get_lm(rnnlm, rnnlm_conf, char_list):
    lm_args = get_model_conf(rnnlm, rnnlm_conf)
    lm_model_module = getattr(lm_args, "model_module", "default") # transformer
    lm_class = dynamic_import_lm(lm_model_module, lm_args.backend) # pytorch
    lm = lm_class(len(char_list), lm_args)
    torch_load(rnnlm, lm)

    return lm

def read_tokens(text, label_dict):
    """Read tokens as a sequence of sentences

    /espnet/espnet/lm/lm_utils.py

    :param text : The LM input
    :param dict label_dict : dictionary that maps token label string to its ID number
    :return list of ID sequences
    :rtype list
    """

    unk = label_dict["<unk>"]
    input_text = " ".join([c if c != " " else "<space>" for c in text.strip()])
    data = np.array([label_dict.get(label, unk) for label in input_text.strip().split()], dtype=np.int32)

    return data

def get_parallel_sentence(text, sos_index, eos_index, device):
    """
       Based on the code espnet/espnet/lm/lm_utils.ParallelSentenceIterator
    """
    lm_input = (torch.from_numpy(np.append([sos_index], text)).unsqueeze(0).to(device),
                torch.from_numpy(np.append(text, [eos_index])).unsqueeze(0).to(device),
               )

    return lm_input

def get_logp(lm, lm_input):
    lm.eval()

    with torch.no_grad():
        # x, t = concat_examples(lm_input, device=device, padding=(0, -100))
        # print(x, " || ", t)
        x, t = lm_input
        loss, logp, n = lm(x,t)

    return logp.item()

def get_new_lm(cuda, corpus):
    torch.manual_seed(7)
    device = torch.device("cuda:"+str(cuda))

    lm_path = "../../models/" + corpus + "/lm"
    if corpus in ["LRS2-BBC", "LRS3-TED"]:
        char_list = ["<blank>", "<unk>", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<space>", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "<eos>"]
    elif corpus in ["VLRF", "CMU-MOSEAS-Spanish", "Multilingual-TEDx-Spanish", "LIP-RTVE"]:
        char_list = ["<blank>", "<unk>", "<space>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "á", "é", "í", "ñ", "ó", "ú", "ü", "<eos>"]
    char_list_dict = {k: v for v, k in enumerate(char_list)}

    lm = get_lm(lm_path+".pth", lm_path+".json", char_list).to(device)
    # print(lm)
    # print("Num parameters:", sum([param.nelement() for param in lm.parameters()]))

    return lm, char_list_dict, device

def get_new_lm_score(lm, text, char_list_dict, device):
    # Get special label IDs
    unk = char_list_dict["<unk>"]
    eos = char_list_dict["<eos>"]

    # Load dataset: read tokens as a sequence of sentences
    input_text = read_tokens(text, char_list_dict)

    lm_input = get_parallel_sentence(input_text, char_list_dict["<eos>"], char_list_dict["<eos>"], device)
    logp = get_logp(lm, lm_input)

    return logp
