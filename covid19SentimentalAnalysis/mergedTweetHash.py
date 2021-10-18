import pandas as pd

df = pd.read_csv('allMerged.csv', index_col=False)
df = df[['Tweet_text','hashtags']]

# df['Tweet_text'] = df['Tweet_text'].str.lower()
# df['hashtags'] = df['hashtags'].str.lower()

df['Tweet_text_merged'] = df.Tweet_text.astype(str).str.cat(df.hashtags.astype(str), sep=' ')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df['Tweet_text_merged'].head())

# df.to_csv('lower_case.csv',index=False)