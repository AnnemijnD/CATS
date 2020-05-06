import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models.patient import Patient

callfile = "data/Train_call.txt"
clinicalfile = "data/Train_clinical.txt"
dfcall = pd.read_csv(callfile, delimiter="\t")
dfclin = pd.read_csv(clinicalfile, delimiter="\t")

# print(dfcall.transpose())
# print(dfclin)
dfcall = dfcall.transpose()
# print(dfcall.head())
# print(dfcall)
# df.loc[indices,'A'] = 16
dfcall["Subgroup"] = 0

dict = {}
for index1, row1 in dfclin.iterrows():
    # print(row1["Subgroup"])
    if row1["Subgroup"] in dict:
        dict[row1["Subgroup"]] += 1
    else:
        dict[row1["Subgroup"]] = 0
print(dict)
#     # print(index1)
#
#     for index2, row2 in dfclin.iterrows():
#         if index1 == row2["Sample"]:
#             # print(row2["Sample"])
#             dfcall.loc[index1, "Subgroup"] = row2["Subgroup"]
#
#             continue
#
# # print(dfcall)
# patients = []
# exceptions = ["Chromosome", "Start","End", "Nclone"]
# for index1, row1 in dfclin.iterrows():
#     if index1 in exceptions:
#         continue
#     else:
#
#         patient = Patient(index1, row1["Subgroup"])
#         patients.append(patient)
#         chromosomedict = {}
#
# for label, content in dfcall.iteritems():
#     print(content)
#
#
#
# # for patient in patients:
# #     print(patient)
#
#
#
#
# # for index, row in dfcall.iterrows():
# #     print(row)
#
data_url = 'http://bit.ly/2cLzoxH'
gapminder = pd.read_csv(data_url)
print(gapminder.head(3))
df1 = gapminder[['continent', 'year','lifeExp']]
print(df1.head())
#   continent  year  lifeExp
# 0      Asia  1952   28.801
# 1      Asia  1957   30.332
# 2      Asia  1962   31.997
# 3      Asia  1967   34.020
# 4      Asia  1972   36.088

# pandas pivot
heatmap1_data = pd.pivot_table(df1, values='lifeExp',
                     index=['continent'],
                     columns='year')
sns.heatmap(heatmap1_data, cmap="YlGnBu")

plt.show()
