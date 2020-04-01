import pandas as pd


callfile = "data/Train_call.txt"
clinicalfile = "data/Train_clinical.txt"
dfcall = pd.read_csv(callfile, delimiter="\t")
dfclin = pd.read_csv(clinicalfile, delimiter="\t")

print(dfcall)
print(dfclin)
