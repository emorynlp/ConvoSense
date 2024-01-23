import os, json, csv
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_chisquare
import matplotlib.pyplot as plt
import scipy.stats as stats
from eval.analysis_utils import *

reasonability_coarse_labels = {
    '4': 1,
    '3': 1,
    '2': 0,
    '1': 0
}

novelty_coarse_labels = {
    '3': 1,
    '2': 1,
    '1': 0
}

novelty_fine_labels = {
    '3': 3,
    '2': 2,
    '1': 1
}

def convert_csv_to_pd(
        dir, 
        sourcepath, 
        only_first=False, 
        reasonability_labels=reasonability_coarse_labels, 
        novelty_labels=novelty_coarse_labels
):
    """
    Load evaluation results from csv for human evaluation of models
    :param to_labels: conversion of label text into label number (`fine_labels` or `coarse_labels`)
    :param criteria_type: the evaluation criteria (`reasonability` or `novelty`)
    :return: DataFrame - item x labeler: label
    """
    source_data = json.load(open(sourcepath))
    reasonability_results = {}
    novelty_results = {}
    group_counts = {}
    columns = []
    for file in os.listdir(dir):
        if '.csv' in file:
            reader = csv.reader(open(f'{dir}/{file}'))
            columns.append(file[:file.index('_')])
            for i, row in enumerate(reader):
                if i > 0 and row[3] and row[4] and row[5]:
                    row_id = f'task{i}'
                    source = source_data[i-1][4].get('model_settings-name', 'n/a')
                    type = row[1]
                    groups = row[3]
                    answers = [x[x.index(')')+2:] for x in row[2].split('\n\n')]
                    if '\n' in groups:
                        groups = [x.strip() for x in groups.split('\n')]
                    elif ',' in groups:
                        groups = [x.strip() for x in groups.split(',')]
                    else:
                        raise Exception(f"problem in parsing groups:\n{groups}")
                    unique_groups = len(set(groups))
                    group_counts.setdefault((type, source, f"{row_id}", len(answers)), []).append(unique_groups)
                    reasonability = row[4]
                    if '\n' in reasonability:
                        reasonability = [x.strip() for x in reasonability.split('\n')]
                    elif ',' in reasonability:
                        reasonability = [x.strip() for x in reasonability.split(',')]
                    else:
                        raise Exception(f"problem in parsing reasonability:\n{reasonability}")
                    novelty = row[5]
                    if '\n' in novelty:
                        novelty = [x.strip() for x in novelty.split('\n')]
                    elif ',' in novelty:
                        novelty = [x.strip() for x in novelty.split(',')]
                    else:
                        raise Exception(f"problem in parsing novelty:\n{novelty}")
                    assert len(reasonability) == len(novelty)
                    for j, (r,n,a) in enumerate(zip(reasonability, novelty, answers)):
                        if not only_first or j == 0:
                            reasonability_results.setdefault((type, source, f"{row_id}_{j}",row[0], a), []).append(reasonability_labels[r])
                            novelty_results.setdefault((type, source, f"{row_id}_{j}", row[0], a), []).append(novelty_labels[n])
    reasonability_df = pd.DataFrame.from_records(
        [(*k, *v) for k, v in reasonability_results.items()],
        columns=['type', 'source', 'task_id', 'context', 'answer', *columns]
    )
    novelty_df = pd.DataFrame.from_records(
        [(*k, *v) for k, v in novelty_results.items()],
        columns=['type', 'source', 'task_id', 'context', 'answer', *columns]
    )
    group_counts_df = pd.DataFrame.from_records(
        [(*k, *v) for k, v in group_counts.items()],
        columns=['type', 'source', 'task_id', 'inferences', *columns]
    )
    return reasonability_df, novelty_df, group_counts_df, columns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def coarse_results_for_models():
    reasonability_df, novelty_df, group_counts_df, label_columns = convert_csv_to_pd(
        'eval/human/models/annotations/',
        'eval/human/models/lab_testing_humanmonobs_humanmonodbs_gptmonodbs.json',
        only_first=False
    )

    print('REASONABILITY')
    print()
    model_reasonabilities = reasonability_df.groupby('source').mean(numeric_only=True)
    print()
    print(model_reasonabilities)

    def count_ones(series):
        df = pd.DataFrame(series)
        counts = df[df['annotator1'] == 1]
        return len(counts)

    def count_total(series):
        df = pd.DataFrame(series)
        df = df.dropna(axis=0)
        return len(df)

    model_reasonability_counts = reasonability_df.groupby('source').apply(count_ones)
    gptdbs_r_prop = model_reasonability_counts.filter(items=['best_gpt_mono_dbs'], axis=0)[0]
    hum_r_prop = model_reasonability_counts.filter(items=['best_human_mono_bs'], axis=0)[0]
    humdbs_r_prop = model_reasonability_counts.filter(items=['best_human_mono_dbs'], axis=0)[0]

    model_reasonability_n = reasonability_df.groupby('source').apply(count_total)
    gptdbs_n = model_reasonability_n.filter(items=['best_gpt_mono_dbs'], axis=0)[0]
    hum_n = model_reasonability_n.filter(items=['best_human_mono_bs'], axis=0)[0]
    humdbs_n = model_reasonability_n.filter(items=['best_human_mono_dbs'], axis=0)[0]

    print()
    gptdbs_vs_humdbs = proportions_chisquare(count=[gptdbs_r_prop, humdbs_r_prop], nobs=[gptdbs_n, humdbs_n])
    print(f"gptdbs vs humdbs p-value: {gptdbs_vs_humdbs[1]}")
    gptdbs_vs_hum = proportions_chisquare(count=[gptdbs_r_prop, hum_r_prop], nobs=[gptdbs_n, hum_n])
    print(f"gptdbs vs hum p-value: {gptdbs_vs_hum[1]}")

    print()
    print()
    print('NOVELTY')
    print()
    model_novelties = novelty_df.groupby('source').mean(numeric_only=True)
    print()
    print(model_novelties)

    model_novelty_counts = novelty_df.groupby('source').apply(count_ones)
    gptdbs_r_prop = model_novelty_counts.filter(items=['best_gpt_mono_dbs'], axis=0)[0]
    hum_r_prop = model_novelty_counts.filter(items=['best_human_mono_bs'], axis=0)[0]
    humdbs_r_prop = model_novelty_counts.filter(items=['best_human_mono_dbs'], axis=0)[0]

    model_novelty_n = novelty_df.groupby('source').apply(count_total)
    gptdbs_n = model_novelty_n.filter(items=['best_gpt_mono_dbs'], axis=0)[0]
    hum_n = model_novelty_n.filter(items=['best_human_mono_bs'], axis=0)[0]
    humdbs_n = model_novelty_n.filter(items=['best_human_mono_dbs'], axis=0)[0]

    print()
    gptdbs_vs_humdbs = proportions_chisquare(count=[gptdbs_r_prop, humdbs_r_prop], nobs=[gptdbs_n, humdbs_n])
    print(f"gptdbs vs humdbs p-value: {gptdbs_vs_humdbs[1]}")
    gptdbs_vs_hum = proportions_chisquare(count=[gptdbs_r_prop, hum_r_prop], nobs=[gptdbs_n, hum_n])
    print(f"gptdbs vs hum p-value: {gptdbs_vs_hum[1]}")

    print()
    print()
    print('Group Counts')
    group_counts = group_counts_df.groupby('source').mean(numeric_only=True)
    print(group_counts)

    group_counts_arr = group_counts_df.groupby('source')
    gptdbs_groups = group_counts_arr.get_group('best_gpt_mono_dbs')['annotator1']
    hum_groups = group_counts_arr.get_group('best_human_mono_bs')['annotator1']
    humdbs_groups = group_counts_arr.get_group('best_human_mono_dbs')['annotator1']

    t_stat, p_value = stats.ttest_ind(gptdbs_groups, hum_groups)
    print(f"gpt-dbs vs hum-bs: {t_stat:.3f} {p_value:.3f}")
    t_stat, p_value = stats.ttest_ind(gptdbs_groups, humdbs_groups)
    print(f"gpt-dbs vs hum-dbs: {t_stat:.3f} {p_value:.3f}")
 
def coarse_results_for_gpt():
    reasonability_df, novelty_df, group_counts_df, label_columns = convert_csv_to_pd(
        'eval/human/convosense/annotations/',
        'eval/human/convosense/lab_testing_convosense.json',
        only_first=False
    )

    print('REASONABILITY')
    print()
    model_reasonabilities = reasonability_df.groupby('source').mean(numeric_only=True)
    print()
    print(model_reasonabilities)

    def count_ones(series):
        df = pd.DataFrame(series)
        counts = df[df['annotator1'] == 1]
        return len(counts)

    def count_total(series):
        df = pd.DataFrame(series)
        df = df.dropna(axis=0)
        return len(df)

    model_reasonability_counts = reasonability_df.groupby('source').apply(count_ones)
    model_reasonability_n = reasonability_df.groupby('source').apply(count_total)

    print()
    print()
    print('NOVELTY')
    print()
    model_novelties = novelty_df.groupby('source').mean(numeric_only=True)
    print()
    print(model_novelties)

    model_novelty_counts = novelty_df.groupby('source').apply(count_ones)
    model_novelty_n = novelty_df.groupby('source').apply(count_total)

    print()
    print()
    print('Group Counts')
    group_counts_df['proportion'] = group_counts_df['annotator1'] / group_counts_df['inferences']
    group_counts = group_counts_df.groupby('source').mean(numeric_only=True)
    print(group_counts)

    group_counts_arr = group_counts_df.groupby('source')
    
    # detailedness

    _, novelty_df, _, _ = convert_csv_to_pd(
        'eval/human/convosense/annotations/',
        'eval/human/convosense/lab_testing_convosense.json',
        only_first=False,
        novelty_labels=novelty_fine_labels
    )

    sums_all_ls = get_sums(novelty_df.drop(['task_id', 'type'], axis='columns').groupby('source'))
    typeless_indices = [(s, c) for s in novelty_df['source'].unique().tolist() for c in novelty_fine_labels.values()]
    sums_perc_all = get_percentages(sums_all_ls, indices=typeless_indices)
    print(sums_perc_all)
    print()

    normalized = sums_perc_all[[2,3]].div(sums_perc_all[[2,3]].sum(axis=1), axis=0).reset_index()
    normalized.columns = ['source'] + list(normalized.columns)[1:]
    normalized['source'] = pd.Categorical(normalized['source'], categories=model_names.keys(), ordered=True)
    normalized = normalized.sort_values('source')
    print(normalized)
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
   return df['annotator1'].value_counts()

model_names = {
    'best_gpt_mono_dbs': 'ConvoSenseM*',
    'best_human_mono_dbs': 'HumanGenM*',
    'best_human_mono_bs': 'HumanGenM'
}

def model_detailedness():
    # detailedness
    reasonability_df, novelty_df, group_counts_df, label_columns = convert_csv_to_pd(
        'eval/human/models/annotations/',
        'eval/human/models/lab_testing_humanmonobs_humanmonodbs_gptmonodbs.json',
        only_first=False,
        novelty_labels=novelty_fine_labels
    )
    # distributions over all types
    sums_all_ls = get_sums(novelty_df.drop(['task_id', 'type'], axis='columns').groupby('source'))
    typeless_indices = [(s, c) for s in novelty_df['source'].unique().tolist() for c in novelty_fine_labels.values()]
    sums_perc_all = get_percentages(sums_all_ls, indices=typeless_indices)
    print(sums_perc_all)
    print()

    normalized = sums_perc_all[[2,3]].div(sums_perc_all[[2,3]].sum(axis=1), axis=0).reset_index()
    normalized.columns = ['source'] + list(normalized.columns)[1:]
    normalized['source'] = pd.Categorical(normalized['source'], categories=model_names.keys(), ordered=True)
    normalized = normalized.sort_values('source')
    print(normalized)
    print()

    models = list(normalized.source.values)
    x_pos = np.arange(len(models))
    width = 0.75

    fig, ax = plt.subplots(layout='tight', figsize=(6,6))
    colorings = ['lightsteelblue', 'lightgreen', 'lightgreen']
    plot_elements = []
    measurement = list(normalized[3].values)
    rects = ax.bar(x_pos, measurement, width, label=models, color=colorings)
    plot_elements.append(rects)
    ax.bar_label(rects, labels=[f'{x:.2f}' for x in measurement], padding=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticks(x_pos, [model_names[m] for m in models])
    plt.tick_params(bottom=False)
    # ax.get_legend().remove()
    ax.set_ylim(0, 0.80)

    # Show the plot
    plt.show()



if __name__ == '__main__':

    # get reasonability, novelty, and diversity results for model inferences
    print('## models ##')
    coarse_results_for_models()

    # get detailedness of model inferences
    model_detailedness()

    print()
    print('##################################')
    print()

    # get reasonability, novelty, detailedness, and diversity results for ConvoSense data
    print('## convosense ##')
    coarse_results_for_gpt()