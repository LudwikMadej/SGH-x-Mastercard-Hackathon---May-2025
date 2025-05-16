from Merging import Merger
from toolkit import *
df=load_data()
merger = Merger(df_base=df)

df = merger.mergeAndTransform()

df.sort_values(by='timestamp', ascending=True,inplace=True)
train_split=df.iloc[:400000,:]
test_split=df.iloc[400000:,:]
for i in range(8):
    df_ = train_split.iloc[i*50_000:(i+1)*50_000, :]
    df_.to_csv(f"../Data/Preprocessed_Data/train/train_{i+1:02}.csv", index=False)

for i in range(2):
    df_ = test_split.iloc[i*50_000:(i+1)*50_000, :]
    df_.to_csv(f"../Data/Preprocessed_Data/test/test_{i+1:02}.csv", index=False)
