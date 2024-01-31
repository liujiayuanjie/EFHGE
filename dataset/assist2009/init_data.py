import pandas as pd
import csv
import argparse
# 'user_id', 'problem_id', 'correct', 'order_id' template_id skill_id
def write_csv(path, rows):
    csv_file = open(path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerows(rows)
    csv_file.close()

def init_data(data_path):
    data = pd.read_csv(data_path, sep = ',', low_memory = False)
    data = data.sort_values(['order_id'])
    data = data[~data['skill_name'].isna()]

    counts = data['user_id'].value_counts()
    lrns = counts[counts > 1].index.tolist()
    data = data[data['user_id'].isin(lrns)]

    qsts = list(set(data['problem_id'].tolist()))
    qsts = [int(qst) for qst in qsts]
    qsts.sort()
    qsts = [[qst, i] for i, qst in enumerate(qsts)]
    write_csv('./qst.csv', qsts)
    qsts = dict(qsts)

    usrs = list(set(data['user_id'].tolist()))
    usrs = [int(usr) for usr in usrs]
    usrs.sort()
    usrs = [[usr, i] for i, usr in enumerate(usrs)]
    write_csv('./usr.csv', usrs)
    usrs = dict(usrs)

    qst_skl = {}
    for qst, skl in data[['problem_id', 'skill_id']].values.tolist():
        qst = qsts[qst]
        if qst not in qst_skl:
            qst_skl[qst] = set()
        qst_skl[qst].add(skl)
    
    for qst in qst_skl:
        skl = sorted(list(qst_skl[qst]))
        skl = '-'.join([str(int(s)) for s in skl])
        qst_skl[qst] = skl
    
    skls = sorted(list(set([qst_skl[qst] for qst in qst_skl])))
    skls = [[skl, i] for i, skl in enumerate(skls)]
    write_csv('./skl.csv', skls)

    skls = dict(skls)

    qst_skl = [[i, skls[qst_skl[i]]] for i in range(len(qsts))]
    write_csv('./qst_skl.csv', qst_skl)

    events = data[['user_id', 'problem_id', 'correct', 'order_id']].values.tolist()
    events = [[usrs[usr], qsts[qst], cor, order] for usr, qst, cor, order in events]

    write_csv('./event.csv', events)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('--data_path', type = str)
    args, _ = parser.parse_known_args()
    init_data(args.data_path)