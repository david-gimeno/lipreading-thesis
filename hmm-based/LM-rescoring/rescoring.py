import os
import sys
from tqdm import tqdm
from scripts.pytorch_lm import get_new_lm, get_new_lm_score

if __name__ == "__main__":
    database = sys.argv[1]
    njobs = int(sys.argv[2])
    graphcost_dir = sys.argv[3]
    hyp_dir = sys.argv[4]
    dst_dir = sys.argv[5]

    lm, char_list_dict, device = get_new_lm(0, database)

    for njob in range(1, njobs+1):
        graphcost_path = os.path.join(graphcost_dir, str(njob)+".graphcost")
        hyp_path = os.path.join(hyp_dir, str(njob)+".hyp")

        graphcosts = {l.split()[0]:float(l.split()[1].strip()) for l in open(graphcost_path, "r").readlines()}
        hyps = {l.split()[0]:" ".join(l.split()[1:]).strip() for l in open(hyp_path, "r").readlines()}


        dst_path = os.path.join(dst_dir, str(njob)+".newlmcost")
        with open(dst_path, "w") as f:
            for key in tqdm(graphcosts.keys()):
                hyp_text = hyps[key]
                graphcost = graphcosts[key]

                new_lm_logp = get_new_lm_score(lm, hyp_text, char_list_dict, device)
                new_lm_score = graphcost + new_lm_logp

                f.write(key + " " + str(new_lm_score) + "\n")

