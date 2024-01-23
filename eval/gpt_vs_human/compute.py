import random
random.seed(818)
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
from eval.analysis_utils import *

fine_labels = {
    'reasonability': {
        'always/likely': 4,
        'sometimes/possible': 3,
        'never/farfetched': 2,
        'invalid/nonsense': 1
    },
    'novelty': {
        'very novel': 3,
        'novel': 2,
        'redundant': 1,
        'new & detailed': 3,
        'new & simple': 2,
        'purely repetitive': 1
    }
}

coarse_labels = {
    'reasonability': {
        'always/likely': 2,
        'sometimes/possible': 2,
        'never/farfetched': 1,
        'invalid/nonsense': 1
    },
    'novelty': {
        'very novel': 2,
        'novel': 2,
        'redundant': 1,
        'new & detailed': 2,
        'new & simple': 2,
        'purely repetitive': 1
    }
}

granularity_labels = {
    'fine': fine_labels,
    'coarse': coarse_labels
}

def mc_to_pd(dir, to_labels, criteria_type):
    """
    Load evaluation results (multiple choice) per criteria
    :param to_labels: conversion of label text into label number (`fine_labels` or `coarse_labels`)
    :param criteria_type: the evaluation criteria (`reasonability` or `novelty`)
    :return: DataFrame - item x labeler: label
    """
    results = {}
    for file in os.listdir(dir):
        if '.json' in file:
            with open(f'{dir}/{file}') as f:
                data = json.load(f)
            for r in data:
                type = r["type"]
                task_id = r["task_id"]  # id for multiple annotations of single item
                for a in ['a1', 'a2']:
                    source = r[f"{a}source"]
                    results.setdefault((type, source, task_id), []).append(to_labels[r[f"{a}-{criteria_type}"]])
    df = pd.DataFrame.from_records(
        [(*k, *v) for k, v in results.items()],
        columns=['type', 'source', 'task_id', 'label1', 'label2']
    )
    return df

def mc_to_pd_item(dir, to_labels, all=True):
    """
    Load evaluation results (multiple choice) per item
    :param to_labels: conversion of label text into label number
    :param all: whether to include all doubly-annotated data, or randomly select singly annotated data from it
    :return: DataFrame - item x criteria: label
    """
    selected_tracking = {}
    results = {}
    for file in os.listdir(dir):
        if '.json' in file:
            with open(f'{dir}/{file}') as f:
                data = json.load(f)
            for r in data:
                type = r["type"]
                task_id = r["task_id"]  # id for multiple annotations of single item
                selected = True
                if not all:
                    # randomly select singly annotated data
                    if task_id not in selected_tracking:
                        # case 1: randomly choose
                        selected = random.choice([True, False])
                        selected_tracking[task_id] = selected
                    elif not selected_tracking[task_id]:
                        # case 2: second item was chosen, this is the second item
                        selected = True
                    else:
                        # case 3: first item was chosen, this is the second item
                        selected = False
                if selected:
                    for a in ['a1', 'a2']:
                        source = r[f"{a}source"]
                        results.setdefault((type, source, task_id, r["task_response_id"], r["context"], r[a]), []).extend(
                            [item for gran in ['fine', 'coarse'] for item in [
                                to_labels[gran]['reasonability'][r[f"{a}-reasonability"]],
                                to_labels[gran]['novelty'][r[f"{a}-novelty"]]
                            ]]
                        )
    df = pd.DataFrame.from_records(
        [(*k, *v) for k, v in results.items()],
        columns=['type', 'source', 'task_id', 'task_response_id', 'context', 'inference', 'reasonabilityf', 'noveltyf', 'reasonabilityc', 'noveltyc']
    )
    return df



def interannotator_agreement(inputs):
    """
    Calculate and display interannotator agreement (raw, ka, ac1/2)
    """
    for criteria in ['reasonability', 'novelty']:
        for granularity in ['coarse']:
            print('#'*50)
            print(f'{criteria} - {granularity}')
            print('#' * 50)
            print()
            if isinstance(inputs, str):
                criteria_labels = mc_to_pd(dir=inputs, to_labels=granularity_labels[granularity][criteria], criteria_type=criteria)
            else:
                criteria_labels = inputs
            categories = list(set(granularity_labels[granularity][criteria].values()))
            # by type
            ka = criteria_labels.groupby('type')[[f'label1', f'label2']].apply(krippendorfs_alpha, ci=False, level='ordinal')
            raw = criteria_labels.groupby('type')[[f'label1', f'label2']].apply(raw_agreement)
            gwet = criteria_labels.groupby('type')[[f'label1', f'label2']].apply(gwetac, level='ordinal', categories=categories) # https://www.agreestat.com/book4/9780970806284_chap3.pdf
            cohen = criteria_labels.groupby('type')[[f'label1', f'label2']].apply(cohenk, categories=categories)
            # all types together
            all_data = criteria_labels[[f'label1', f'label2']]
            ka_all = krippendorfs_alpha(all_data, ci=False, level='ordinal')
            raw_all = raw_agreement(all_data)
            gwet_all = gwetac(all_data, level='ordinal', categories=categories)
            cohen_all = cohenk(all_data, categories=categories)
            all_df = pd.DataFrame({'raw-agree': raw_all, 'gwet': gwet_all, 'cohen': cohen_all, 'k-alpha': ka_all["Krippendorff's alpha"].item(), 'n': ka_all["n"].item()}, index=['all'])
            # one dataframe
            results = pd.concat([raw, gwet, cohen, ka], axis='columns')
            results.columns = ['raw-agree', 'gwet', 'cohen', 'k-alpha', 'n']
            results = pd.concat([all_df, results], axis='index')
            print(results)
            print()

def get_percentages(*sums, indices):
    totals = {}
    for index in indices:
        values = [s[index] if index in s.index else 0 for s in sums]
        category_idx = len(index) - 1
        totals.setdefault(index[:category_idx], {})[index[category_idx]] = sum(values)
    sums = pd.DataFrame(totals).T
    sums['n'] = sums.sum(axis='columns')
    sums_perc = sums.div(sums['n'], axis='rows')
    sums_perc['n'] = sums['n']
    sums_perc = sums_perc.round(2)
    return sums_perc

def get_sums(df):
   return df['label'].value_counts()

def human_evaluation_stat_test(dir, runs=1):
    """
    Statistical testing of coarse-grained criteria evaluation for each dataset
    """

    results = {}

    for i in range(runs):
        for all in [False]:
            criteria_pd = mc_to_pd_item(dir=dir, to_labels=granularity_labels, all=all)
            for criteria in ['reasonability', 'novelty']:
                for granularity in ['coarse']:
                    results.setdefault(criteria, {}).setdefault(granularity, {})
                    categories = list(set(granularity_labels[granularity][criteria].values()))
                    sources = criteria_pd['source'].unique().tolist()
                    types = criteria_pd['type'].unique().tolist()
                    indices = [(t, s, c) for t in types for s in sources for c in categories]
                    # all datapoints
                    # distributions per type
                    criteria_labels = criteria_pd[['type', 'source', 'task_id', f'{criteria}{granularity[0]}']]
                    criteria_labels.columns = ['type', 'source', 'task_id', 'label']
                    sums_ls = get_sums(criteria_labels.drop(['task_id'], axis='columns').groupby(['type', 'source']))
                    sums_perc = get_percentages(sums_ls, indices=indices)
                    # distributions over all types
                    sums_all_ls = get_sums(criteria_labels.drop(['task_id', 'type'], axis='columns').groupby('source'))
                    typeless_indices = [(s, c) for s in sources for c in categories]
                    sums_perc_all = get_percentages(sums_all_ls, indices=typeless_indices)

                    # paired datapoints
                    for s in sources:
                        if 'gpt' not in s:
                            # distributions per type
                            task_ids_in_s = criteria_labels[criteria_labels['source'] == s]['task_id']
                            paired = criteria_labels[criteria_labels['task_id'].isin(task_ids_in_s)]
                            sums_ls = get_sums(paired.drop('task_id', axis='columns').groupby(['type', 'source']))
                            sums_perc = get_percentages(sums_ls, indices=[x for x in indices if x[1] in {'gpt', s}])
                            sums_perc = sums_perc.dropna()
                            # distributions over all types
                            sums_all_ls = get_sums(paired.drop(['task_id', 'type'], axis='columns').groupby('source'))
                            typeless_indices = [(s, c) for s in {'gpt', s} for c in categories]
                            sums_perc_all = get_percentages(sums_all_ls, indices=typeless_indices)
                            sums_perc_all = sums_perc_all.dropna()
                            if not all and granularity == 'coarse':
                                # statistical testing for COARSE evaluations ONLY

                                sums_perc = sums_perc.reset_index()
                                reasonable = sums_perc.pivot(index='level_0', columns='level_1', values=2)
                                reasonable.index = [
                                    x[:3] + x.replace('_o', '$_o$').replace('_s', '$_s$')[-4:]
                                    if '_' in x else x[:3]
                                    for x in reasonable.index
                                ]
                                reasonable = reasonable[[s, 'gpt']]
                                reasonable = reasonable.sort_values('gpt', ascending=True)

                                # 2x2 contigency table
                                other_labels = paired[paired['source'] == s].set_index('task_id').drop(['type', 'source'], axis='columns')
                                other_labels.columns = [s]
                                gpt_labels = paired[paired['source'] == 'gpt'].set_index('task_id').drop(['type', 'source'], axis='columns')
                                gpt_labels.columns = ['gpt']
                                joined = pd.concat([gpt_labels, other_labels], axis='columns')
                                both_success = joined[(joined['gpt'] == joined[s]) & (joined['gpt'] == 2)]
                                both_fail = joined[(joined['gpt'] == joined[s]) & (joined['gpt'] == 1)]
                                gpt_win = joined[(joined['gpt'] != joined[s]) & (joined['gpt'] == 2)]
                                other_win = joined[(joined['gpt'] != joined[s]) & (joined['gpt'] == 1)]
                                contingencies = [
                                        (len(both_success), len(gpt_win)),
                                        (len(other_win), len(both_fail))
                                    ]
                                contingency_table = pd.DataFrame.from_records(
                                    contingencies,
                                    index=['gpt-success', 'gpt-fail'],
                                    columns=[f'{s}-success', f'{s}-fail']
                                )
                                total = len(both_success) + len(both_fail) + len(gpt_win) + len(other_win)
                                gpt_dp = len(gpt_win) / total
                                other_dp = len(other_win) / total
                                result = mcnemar(contingencies, exact=False, correction=False)

                                results[criteria][granularity].setdefault(s, []).append(
                                    (sums_perc_all.iloc[:, 1], gpt_dp, other_dp, result.pvalue)
                                )
    return results

def human_evaluation_novelty_finegrained(dir, runs=1):
    """
    Results from human evaluation of fine-grained novelty 
    """

    results = {}

    for i in range(runs):
        for all in [False]:
            criteria_pd = mc_to_pd_item(dir=dir, to_labels=granularity_labels, all=all)
            for criteria in ['novelty']:
                for granularity in ['fine']:
                    results.setdefault(criteria, {}).setdefault(granularity, {})
                    categories = list(set(granularity_labels[granularity][criteria].values()))
                    sources = criteria_pd['source'].unique().tolist()
                    types = criteria_pd['type'].unique().tolist()
                    indices = [(t, s, c) for t in types for s in sources for c in categories]
                    # all datapoints
                    # distributions per type
                    criteria_labels = criteria_pd[['type', 'source', 'task_id', f'{criteria}{granularity[0]}']]
                    criteria_labels.columns = ['type', 'source', 'task_id', 'label']
                    sums_ls = get_sums(criteria_labels.drop(['task_id'], axis='columns').groupby(['type', 'source']))
                    sums_perc = get_percentages(sums_ls, indices=indices)
                    # distributions over all types
                    sums_all_ls = get_sums(criteria_labels.drop(['task_id', 'type'], axis='columns').groupby('source'))
                    typeless_indices = [(s, c) for s in sources for c in categories]
                    sums_perc_all = get_percentages(sums_all_ls, indices=typeless_indices)

                    # paired datapoints
                    for s in sources:
                        if 'gpt' not in s:
                            # distributions per type
                            task_ids_in_s = criteria_labels[criteria_labels['source'] == s]['task_id']
                            paired = criteria_labels[criteria_labels['task_id'].isin(task_ids_in_s)]
                            sums_ls = get_sums(paired.drop('task_id', axis='columns').groupby(['type', 'source']))
                            sums_perc = get_percentages(sums_ls, indices=[x for x in indices if x[1] in {'gpt', s}])
                            sums_perc = sums_perc.dropna()
                            # distributions over all types
                            sums_all_ls = get_sums(paired.drop(['task_id', 'type'], axis='columns').groupby('source'))
                            typeless_indices = [(s, c) for s in {'gpt', s} for c in categories]
                            sums_perc_all = get_percentages(sums_all_ls, indices=typeless_indices)
                            sums_perc_all = sums_perc_all.dropna()
                            
                            results[criteria][granularity].setdefault(s, []).append(
                                sums_perc_all
                            )
    
    # aggregate over all runs
    print('\nAggregated:')
    pairs = []
    stats = { # (detailed, simple)
        'gpt': [], 
        'other': [],
    }
    for source in sources:
        if 'gpt' not in source:
            combined = pd.concat(results['novelty']['fine'][source], axis=0)
            averaged = combined.groupby(combined.index).mean()
            normalized = averaged[[2,3]].div(averaged[[2,3]].sum(axis=1), axis=0).reset_index()
            normalized.columns = ['source'] + list(normalized.columns)[1:]
            normalized['source'] = normalized['source'].apply(lambda x: x[0])
            print(normalized)
            print()
            pairs.append(('GPT', source[0].upper() + source[1:]))
            stats['gpt'].append(normalized[normalized['source'] == 'gpt'][[3,2]].values[0])
            stats['other'].append(normalized[normalized['source'] == source][[3,2]].values[0])

    x_pos = np.arange(len(pairs))
    width = 0.40  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    colorings = ['lightsteelblue', 'lightgreen']
    plot_elements = []
    for j, (attribute, measurements) in enumerate(stats.items()):
        for i in range(len(measurements[0]))[:1]:
            measurement = [x[i]*100 for x in measurements]
            if i == 0:
                y_offset = [i] * len(measurement)
            else:
                y_offset = [x[i-1]*100 for x in measurements]
            offset = width * multiplier
            rects = ax.bar(x_pos + offset, measurement, width, bottom=y_offset, color=colorings[j]) # , label=attribute)
            plot_elements.append(rects)
            # ax.bar_label(rects, labels=[f'{x:.2f}' for x in measurement], padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    xtick_pos = [pos for x in x_pos for pos in [x, x + width]]
    xtick_pos[3] = xtick_pos[3] + 0.05
    ax.set_xticks(xtick_pos, [p if p != 'Comfact' else 'ComFact' for pair in pairs for p in pair])
    plt.tick_params(bottom=False)
    # ax.get_legend().remove()
    ax.set_ylim(0, 0.80)
    plt.xticks(fontsize=16, weight='bold')
    plt.yticks([0,20,40,60,80], fontsize=16, weight='bold')
    ax.tick_params(axis='y', which='both', left=False)

    # Show the plot
    plt.show()


if __name__ == "__main__":

    # interannotator agreements
    print('#'*50)
    print('INTERANNOTATOR AGREEMENTS')
    print('#'*50)
    interannotator_agreement('eval/gpt_vs_human/annotations/')
    print()

    # Statistical test of coarse-grained reasonability and novelty (human judged) of gpt vs human inferences
    # results used in paper found in `eval/gpt_vs_human/gpt_vs_human_100runs.txt`
    print('#'*50)
    print('STATISTICAL TESTING')
    print('#'*50)
    results = human_evaluation_stat_test('eval/gpt_vs_human/annotations/', runs=100)
    for criteria, granularities in results.items():
        print(criteria)
        for granularity, sources in granularities.items():
            for source, statistics in sources.items():
                print(f'\t{source}')
                print(f"\t% of runs where p-value < 0.05: {np.mean([1 if x[3] < 0.05 else 0 for x in statistics])*100:.1f}")
                avg_gpt, std_gpt = np.mean([x[0]['gpt'] for x in statistics]), np.std([x[0]['gpt'] for x in statistics]) # average proportion of gpt inferences
                avg_other, std_other = np.mean([x[0][source] for x in statistics]), np.std([x[0][source] for x in statistics]) # average proportion of other_data inferences
                avg_gpt_dp = np.mean([x[1] for x in statistics])
                avg_other_dp = np.mean([x[2] for x in statistics])
                print(f"\t{avg_gpt*100:.0f} [{std_gpt*100:.2f}] ({avg_gpt_dp:.2f})")
                print(f"\t{avg_other*100:.0f} [{std_other*100:.2f}] ({avg_other_dp:.2f})")
                print()
    print()

    # plot detailedness
    print('#'*50)
    print('DETAILEDNESS')
    print('#'*50)
    human_evaluation_novelty_finegrained('eval/gpt_vs_human/annotations/', runs=100)