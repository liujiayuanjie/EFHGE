import pandas as pd
import numpy as np
import csv
import argparse

def write_csv(path, rows):
    csv_file = open(path, 'w', newline = '')
    writer = csv.writer(csv_file)
    writer.writerows(rows)
    csv_file.close()

def init_data(data_path, lrn_num = None, seed = 0):
    data = pd.read_csv('%s/junyi_problemlog_original.csv' % data_path, sep = ',', low_memory = False)
    data = data.sort_values(['time_done'])

    counts = data['user_id'].value_counts()
    lrns = counts[counts > 1].index.tolist()
    
    if lrn_num:
        np.random.seed(seed)
        np.random.shuffle(lrns)
        lrns = lrns[: lrn_num]

    data = data[data['user_id'].isin(lrns)]

    usrs = list(set(data['user_id'].tolist()))
    usrs.sort()
    usrs = [[usr, i] for i, usr in enumerate(usrs)]
    write_csv('./usr.csv', usrs)
    usrs = dict(usrs)

    exs_data = pd.read_csv('%s/junyi_exercise_table.csv' % data_path, sep = ',', low_memory = False)
    exs_data = exs_data[['name', 'topic', 'area']]
    exs_data = exs_data.values.tolist()

    qsts = list(set([qst for qst, skl, area in exs_data]))
    qsts.sort()
    qsts = [[qst, i] for i, qst in enumerate(qsts)]
    write_csv('./qst.csv', qsts)
    qsts = dict(qsts)

    skls = list(set([str(skl) for qst, skl, area in exs_data]))
    skls.sort()
    skls = [[skl, i] for i, skl in enumerate(skls)]
    write_csv('./skl.csv', skls)
    skls = dict(skls)

    qst_skl = {}
    for qst, skl, area in exs_data:
        qst = qsts[qst]
        qst_skl[qst] = skls[str(skl)]

    qst_skl = [[i, qst_skl[i]] for i in range(len(qsts))]
    write_csv('./qst_skl.csv', qst_skl)

    events = data[['user_id', 'exercise', 'correct', 'time_done']].values.tolist()
    events = [[usrs[usr], qsts[qst], 0 if cor else 1, order] for usr, qst, cor, order in events]

    write_csv('./event.csv', events)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('--data_path', type = str)
    args, _ = parser.parse_known_args()
    init_data(args.data_path, 5000, 0)