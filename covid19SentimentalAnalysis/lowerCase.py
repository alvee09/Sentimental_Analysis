import pandas as pd

df = pd.read_csv('allMerged.csv', index_col=False)
df = df[['Tweet_text','hashtags']]

df['Tweet_text'] = df['Tweet_text'].str.lower()
df['hashtags'] = df['hashtags'].str.lower()

print(df.head())

df.to_csv('lower_case.csv',index=False)