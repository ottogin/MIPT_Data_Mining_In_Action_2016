import pandas as pd
d = pd.read_csv('../Data/pr_trans')
d['week_day'] = d['day'] % 7
d['hour'] = d['time'] % (3600 * 24) / 3600
d.to_csv('../Data/prpr_trans')

ids = d['customer_id'].unique()
cust = [d[d.customer_id == idq] for idq in ids]

total_amount = [sum(c.amount) for c in cust]
total_abs_amount = [sum(map(abs, c.amount)) for c in cust]
av_total_amount = [sum(c.amount) / len(c) for c in cust]
av_abs_amount = [sum(map(abs, c.amount)) / len(c) for c in cust]

nd = ids
nd['total_amount'] = total_amount
nd['total_abs_amount'] = total_abs_amount
nd['av_total_amount'] = av_total_amount
nd['av_abs_amount'] = av_abs_amount
nd = nd.set_index(ids)

nd.to_csv('../Data/homan_trans')






