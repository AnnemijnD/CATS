import pandas as pd


callfile = "data/Train_call.txt"
clinicalfile = "data/Train_clinical.txt"
df = pd.read_csv(callfile, delimiter="  ")

print(df)
