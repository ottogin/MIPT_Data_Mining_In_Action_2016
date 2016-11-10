import pandas as pd
d = pd.read_csv('../Data/pr_trans')

ids = d['customer_id'].unique()
cust = [d[d.customer_id == idq] for idq in ids]

total_amount = [sum(c.amount) for c in cust]
total_abs_amount = [sum(map(abs, c.amount)) for c in cust]
av_total_amount = [sum(c.amount) / len(c) for c in cust]
av_abs_amount = [sum(map(abs, c.amount)) / len(c) for c in cust]

nd = pd.concat([pd.DataFrame({'total_amount': total_amount}),
				pd.DataFrame({'total_abs_amount': total_abs_amount}),
				pd.DataFrame({'av_total_amount': av_total_amount}),
				pd.DataFrame({'av_abs_amount': av_abs_amount})],
				 axis=1)
nd = nd.set_index(ids)
print(nd)
nd.to_csv('../Data/homan_trans', index=True)






