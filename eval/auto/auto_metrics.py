import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
import transformers
import evaluate
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from eval.auto.global_vars import (
    BEAMED_GENERATIONS,
    BLEU_MATRIX, BERTSCORE_MATRIX, EMBED_MATRIX
)
from utils import (
    extract_multianswers,
)
from scipy.optimize import linear_sum_assignment

import random
random.seed(1234)
transformers.set_seed(1234)

# visualize previous results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print('loading metrics...')
BERTSCORE, SACREBLEU, SENTENCEBERT = None, None, None
while BERTSCORE is None or SACREBLEU is None or SENTENCEBERT is None:
    try:
        BERTSCORE = evaluate.load('bertscore')
        SACREBLEU = evaluate.load('sacrebleu')
        SENTENCEBERT = SentenceTransformer('all-mpnet-base-v2')
    except FileNotFoundError as e:
        print(e)
        time.sleep(1)
        continue

print('loaded all metrics!')

####################################################################################################################
#
#  METRICS
#
####################################################################################################################

def _get_predicted(example, modelname, multianswer, only_one_beam, max_out):
    if not multianswer:
        first_answer = example[BEAMED_GENERATIONS][0]
        if '(1) ' in first_answer:
            start_of_second_answer = first_answer.index('; (2)') if '; (2)' in first_answer else len(first_answer)
            first_answer = first_answer[:start_of_second_answer].replace('(1) ', '')
        return [first_answer]
    else:
        # all beams for single-answer models / parse out all answers from first beam for multi-answer models
        all_answers = extract_multianswers(example, modelname=modelname, only_one_beam=only_one_beam)
        if max_out is not None:
            all_answers = all_answers[:max_out]
        return all_answers

def sacrebleu(model, data, multianswer=False, prefix='', max_out=None):
    """
    Sacrebleu
        - calculated using 1-grams to 4-grams
        - If multiple references provided, then the score is the MAX, not average

    Documentation references:
    https://huggingface.co/docs/datasets/how_to_metrics
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py
    https://github.com/mjpost/sacreBLEU -> use None or '' to fill in reference_lists to be same length

    linear_sum_assignment - similar to MultiTalk approach to measuring diversity of generations w.r.t diverse references
    """

    max_bleu = []
    for example in tqdm(data, desc='bleu'):
        if not multianswer:
            refs = example['all_answers']
            pred = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam=True, max_out=max_out)
            assert len(pred) == 1
            if BLEU_MATRIX in example:
                score = example[BLEU_MATRIX][:len(pred)]
            else:
                raise Exception('BLEU_MATRIX needs to be set first!')
            max_bleu.append(np.max(score))
        else:
            refs = example['all_answers']
            preds = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam='polymorphic' in model, max_out=max_out)
            if BLEU_MATRIX in example:
                scores = example[BLEU_MATRIX][:len(preds)]
            else:
                scores = []   # dim: pred x reference
                for pred in preds:
                    scores.append([
                        SACREBLEU.compute(predictions=[pred], references=[ref], lowercase=True)['score']
                        for ref in refs
                    ])
                example[BLEU_MATRIX] = scores
            row_ind, col_ind = linear_sum_assignment(scores, maximize=True)
            best_scores = [scores[row][col] for row, col in list(zip(row_ind, col_ind))]
            avg_bleu = np.mean(best_scores)
            max_bleu.append(avg_bleu)
    weighted_mean_numerator = 0
    weighted_mean_denominator = 0
    if multianswer:
        for example, bleu_score in zip(data, max_bleu):
            ground_truth_len = len(example['all_answers'])
            preds = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam='polymorphic' in model, max_out=max_out)
            predicted_len = len(preds)
            coverage = min(predicted_len, ground_truth_len) / ground_truth_len
            numerator = (bleu_score * coverage) * ground_truth_len
            denominator = ground_truth_len
            weighted_mean_numerator += numerator
            weighted_mean_denominator += denominator
        return {
            f'{prefix}_weighted': weighted_mean_numerator / weighted_mean_denominator if multianswer else None
        }
    return {
        f'{prefix}': np.mean(max_bleu),
    }

def bertscore(model, data, multianswer=False, prefix='', max_out=None):
    """
    picked rank1 bertscore model according to https://github.com/Tiiiger/bert_score, https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0
    """
    bertscoremodel = 'microsoft/deberta-xlarge-mnli'
    max_bertscore = []
    for example in tqdm(data, desc='bertscore'):
        if not multianswer:
            refs = example['all_answers']
            pred = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam=True, max_out=max_out)
            assert len(pred) == 1
            if BERTSCORE_MATRIX in example:
                score = example[BERTSCORE_MATRIX][:len(pred)]
            else:
                raise Exception('BERTSCORE_MATRIX needs to be set first!')
                # score = BERTSCORE.compute(predictions=pred*len(refs), references=refs, model_type=bertscoremodel)
            max_bertscore.append(np.max(score))
        else:
            refs = example['all_answers']
            preds = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam='polymorphic' in model, max_out=max_out)
            if BERTSCORE_MATRIX in example:
                scores = example[BERTSCORE_MATRIX][:len(preds)]
            else:
                preds_in = [pred for pred in preds for ref in refs]
                refs_in = [ref for pred in preds for ref in refs]
                results = BERTSCORE.compute(predictions=preds_in, references=refs_in, model_type=bertscoremodel)
                sublist_size = len(refs)
                scores = [results['f1'][i:i + sublist_size] for i in range(0, len(results['f1']), sublist_size)]  # dim: pred x reference
                example[BERTSCORE_MATRIX] = scores
            row_ind, col_ind = linear_sum_assignment(scores, maximize=True)
            max_bertscore.append(np.mean([scores[row][col] for row, col in list(zip(row_ind, col_ind))]))
    weighted_mean_numerator = 0
    weighted_mean_denominator = 0
    if multianswer:
        for example, bleu_score in zip(data, max_bertscore):
            ground_truth_len = len(example['all_answers'])
            preds = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam='polymorphic' in model, max_out=max_out)
            predicted_len = len(preds)
            coverage = min(predicted_len, ground_truth_len) / ground_truth_len
            numerator = (bleu_score * coverage) * ground_truth_len
            denominator = ground_truth_len
            weighted_mean_numerator += numerator
            weighted_mean_denominator += denominator
        return {
            f'{prefix}_weighted': weighted_mean_numerator / weighted_mean_denominator if multianswer else None
        }
    return {
        f'{prefix}': np.mean(max_bertscore)
    }

def sentencebert(model, data, multianswer=False, prefix='', max_out=None):
    max_sim = []
    for example in tqdm(data, desc='embedding'):
        ref = example['all_answers']
        pred = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam='polymorphic' in model, max_out=max_out)
        if EMBED_MATRIX in example:
            cosine_scores = example[EMBED_MATRIX][:len(pred)]
        else:
            pred_embed = SENTENCEBERT.encode(pred, convert_to_tensor=True)
            ref_embed = SENTENCEBERT.encode(ref, convert_to_tensor=True)
            cosine_scores = cos_sim(pred_embed, ref_embed).cpu().numpy()
            example[EMBED_MATRIX] = cosine_scores
        if not multianswer:
            max_sim.append(np.max(cosine_scores))
        else:
            row_ind, col_ind = linear_sum_assignment(cosine_scores, maximize=True)
            max_sim.append(np.mean([cosine_scores[row][col] for row, col in list(zip(row_ind, col_ind))]))
    weighted_mean_numerator = 0
    weighted_mean_denominator = 0
    if multianswer:
        for example, bleu_score in zip(data, max_sim):
            ground_truth_len = len(example['all_answers'])
            preds = _get_predicted(example, modelname=model, multianswer=multianswer, only_one_beam='polymorphic' in model, max_out=max_out)
            predicted_len = len(preds)
            coverage = min(predicted_len, ground_truth_len) / ground_truth_len
            numerator = (bleu_score * coverage) * ground_truth_len
            denominator = ground_truth_len
            weighted_mean_numerator += numerator
            weighted_mean_denominator += denominator
        return {
            f'{prefix}_weighted': weighted_mean_numerator / weighted_mean_denominator if multianswer else None
        }
    return {
        f'{prefix}': np.mean(max_sim)
    }