import pandas as pd
import numpy as np
import csv
import argparse
from datetime import datetime

def write_csv(path, rows):
    csv_file = open(path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerows(rows)
    csv_file.close()

def init_data(data_path, skl_level, lrn_num = None, seed = 0):
    awrs = pd.read_csv('%s/answer_metadata_task_3_4.csv' % data_path, sep = ',', low_memory = False)
    format = '%Y-%m-%d %H:%M:%S.%f'
    awrdate = awrs[['AnswerId', 'DateAnswered']].values.tolist()
    awrtme = [[int(datetime.strptime(t, format).timestamp()), int(i)] for i, t in awrdate]
    awrtme = sorted(awrtme, key = lambda x: x[0])

    rows = pd.read_csv('%s/train_task_3_4.csv' % data_path, sep = ',', low_memory = False)
    rows = rows[['AnswerId', 'UserId', 'QuestionId', 'IsCorrect']].values.tolist()
    row_dict = dict([[int(i), [usr, qst, cor]] for i, usr, qst, cor in rows])

    usr_set = set()
    qst_set = set()
    for i, usr, qst, cor in rows:
        usr_set.add(usr)
        qst_set.add(qst)

    usrs = sorted(list(usr_set))
    
    if lrn_num:
        usrs = list(usr_set)
        np.random.seed(seed)
        np.random.shuffle(usrs)
        usrs = usrs[: lrn_num]
        usrs = sorted(usrs)
    else:
        usrs = sorted(list(usr_set))

    usrs = [[usr, i] for i, usr in enumerate(usrs)]
    write_csv('./usr.csv', usrs)
    usrs = dict(usrs)

    qsts = sorted(list(qst_set))
    qsts = [[qst, i] for i, qst in enumerate(qsts)]
    write_csv('./qst.csv', qsts)
    qsts = dict(qsts)

    events = []
    for t, awr in awrtme:
        if awr in row_dict:
            usr, qst, cor = row_dict[awr]
            if usr in usrs:
                events.append([usrs[usr], qsts[qst], cor, t])
    write_csv('./event.csv', events)

    qst_info = pd.read_csv('%s/question_metadata_task_3_4.csv' % data_path, sep = ',', low_memory = False)
    qst_info = qst_info.values.tolist()
    qst_dict = dict([[qst, eval(skllst)] for qst, skllst in qst_info])    
    
    skl_set = set([qst_dict[qst][skl_level] for qst in qst_dict])
    skls = sorted(list(skl_set))
    skls = [[skl, i] for i, skl in enumerate(skls)]
    write_csv('./skl.csv', skls)
    skls = dict(skls)

    qst_skl = {}
    for qst in qsts:
        skl = qst_dict[qst][skl_level]
        qst_skl[qsts[qst]] = skls[skl]
    
    qst_skl = [[i, qst_skl[i]] for i in range(len(qsts))]
    write_csv('./qst_skl.csv', qst_skl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('--data_path', type = str)
    args, _ = parser.parse_known_args()
    init_data(args.data_path, 3, 2000)
