import krippendorff
from scipy.stats import bootstrap
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np
from irrCAC.raw import CAC
import os, json
from pprint import pprint

########################################################################################################
# Data loading
########################################################################################################
def pairwise():
    results = {}
    for file in os.listdir('evaluation/pilot_evaluation_0410/'):
        if 'format' not in file and 'gpt' not in file:
            with open(f'evaluation/pilot_evaluation_0410/{file}') as f:
                data = json.load(f)
            for r in data:
                type = r["type"]
                comp = ','.join((r['a1source'], r['a2source'])) if r['a1source'] == 'gpt' else ','.join((r['a2source'], r['a1source']))
                results.setdefault(type, {}).setdefault(comp, {})
                for criteria in ['reasonability  ', 'interestingness']:
                    results[type][comp].setdefault(criteria, [0, 0, 0])
                    result = r[criteria[:-2] if criteria.startswith('rea') else criteria]
                    if 'equal' in result.lower():
                        results[type][comp][criteria][1] += 1
                    else:
                        w = result.replace('{{', '').replace('}}', '')
                        winner = r[f'{w}source']
                        if winner == 'gpt':
                            results[type][comp][criteria][0] += 1
                        else:
                            results[type][comp][criteria][2] += 1
    pprint(results)

def multiple_choice():
    task_ids = []
    results = {}
    for file in os.listdir('evaluation/evaluation_30_0411/'):
        if 'format' not in file and 'gpt' not in file:
            with open(f'evaluation/evaluation_30_0411/{file}') as f:
                data = json.load(f)
            for r in data:
                type = r["type"]
                if r["task_id"] in task_ids:
                    x = 1
                task_ids.append(r["task_id"])
                results.setdefault(type, {})
                for a in ['a1', 'a2']:
                    source = r[f"{a}source"]
                    if source not in results[type]:
                        results[type].setdefault(
                            source,
                            {
                                'reasonability': {
                                    'always/likely': 0,
                                    'sometimes/possible': 0,
                                    'never/farfetched': 0,
                                    'invalid/nonsense': 0
                                },
                                'novelty': {
                                    'very novel': 0,
                                    'novel': 0,
                                    'redundant': 0
                                }
                            }
                        )
                    for c in ['reasonability', 'novelty']:
                        criteria = r[f"{a}-{c}"]
                        results[type][source][c][criteria] += 1
    print(json.dumps(results, indent=2))

########################################################################################################
# Stats
########################################################################################################

class sym:

    def __call__(self):
        return [
            v for k, v in self.__class__.__dict__.items()
            if not k.startswith('__')
        ]

    def __iter__(self):
        return iter(self())

    def __contains__(self, item):
        return item in self()

class stat(sym):
    kripp_alpha = "Krippendorff's alpha"
    ci_low = "CI low"
    ci_high = "CI high"
    n = 'n'

def bootstrap_ci(data, statistic_fn, n_resamples=10**3, confidence_level=0.95):
    wrapped_data = [dict(point=d) for d in data]
    statistic_fn_wrapper = lambda ds: statistic_fn([d['point'] for d in ds])
    result = bootstrap((wrapped_data,), statistic_fn_wrapper, vectorized=False,
                                n_resamples=n_resamples, confidence_level=confidence_level)
    return result.confidence_interval

def krippendorfs_alpha(df, ci=True, level='ordinal'):
    """
    :param df: pandas dataframe: items x labeler: label
    :return:
    """
    ratings = df.to_numpy()
    ka = lambda x: krippendorff.alpha(x.T, level_of_measurement=level)
    try:
        alpha = ka(ratings)
    except AssertionError:
        alpha = None
    if ci:
        try:
            low, high = bootstrap_ci(ratings, lambda x: ka(np.array(x)))
        except AssertionError:
            low, high = None, None
        result = {
            stat.kripp_alpha: alpha,
            stat.ci_low: low, stat.ci_high: high,
            stat.n: len(df)
        }
    else:
        result = {
            stat.kripp_alpha: alpha,
            stat.n: len(df)
        }
    return pd.Series(result.values(), result)

def gwetac(ratings_df, level, categories):
    cac_ratings = CAC(ratings_df, weights=level, categories=categories)
    result = cac_ratings.gwet()
    return result['est']['coefficient_value']

def fleissc(ratings_df, categories):
    cac_ratings = CAC(ratings_df, categories=categories)
    result = cac_ratings.fleiss()
    return result['est']['coefficient_value']

def cohenk(ratings_df, categories):
    result = cohen_kappa_score(ratings_df['label1'], ratings_df['label2'])
    return result

def check_same_values(row: pd.Series):
    # Check if all cells in each row have the same value
    row = row.dropna()
    return (row == row.iloc[0]).all()

def raw_agreement(df):
    sames = df.apply(check_same_values, axis=1)
    percentage_same = sames.mean()
    return percentage_same