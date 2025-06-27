import pandas as pd


# df = pd.read_excel('./datasets/FY2122.xlsx')
df = pd.read_csv('./datasets/FY2324.csv')


# Keep only the first 31 columns
df = df.iloc[:, :31]

df = df.fillna('')
for col in df.columns[6:31]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)


print(df.to_csv(index=False))