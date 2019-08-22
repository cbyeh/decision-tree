import pandas as pd

df = pd.DataFrame([{'c1':10, 'c2':100}, {'c1':11,'c2':110}, {'c1':12,'c2':120}])

# df = pd.DataFrame(dict)
# print(df)

for index, row in df.iterrows():
    print(row[1])
    # print(row['c1'], row['c2'])clear
