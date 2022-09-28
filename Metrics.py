import numpy as np
import os

def compute_batch_total_bleu(captions_ref, captions_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch
    '''
    pathcode = 'Code'
    pathorg = 'datasets/web/EVALUATION_SET'
    ref_list = []
    cand_list = []
    for i in os.listdir(pathorg):
        if i.endswith('.gui'):
            with open(os.path.join(pathorg, i)) as f:
                print(f.readlines())


def grouper(seq, n):
    '''Extract all n-grams from a sequence
    '''
    ngrams = []
    for i in range(len(seq) - n + 1):
        ngrams.append(seq[i:i+n])
    
    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram
    '''
    total_matches = 0
    ngrams_r = grouper(reference, n)
    ngrams_c = grouper(candidate, n)
    total_num = len(ngrams_c)
    assert total_num > 0
    for ngram_c in ngrams_c:
        if ngram_c in ngrams_r:
            total_matches += 1
    return total_matches/total_num
    


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate
    '''
    if len(candidate) == 0:
        return 0
    if len(reference) <= len(candidate):
        return 1
    return np.exp(1 - (len(reference)/len(candidate)))



def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score
    '''
    bp = brevity_penalty(reference, hypothesis)
    prec = 1
    cand_len = min(n, len(hypothesis))
    if(cand_len == 0):
        return 0
    for i in range(1, cand_len + 1):
        prec = prec * n_gram_precision(reference, hypothesis, i)
    prec = prec ** (1/n)
    return bp * prec