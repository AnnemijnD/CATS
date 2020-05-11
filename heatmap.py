import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    # data_url = 'http://bit.ly/2cLzoxH'
    # gapminder = pd.read_csv(data_url)
# print(gapminder.head(3))
df1 = pd.read_csv("results_features_good/heatmap_ReliefF_twotypes.csv")
print(df1.head())
df1 = df1.sort_values(by=['freqs'])
df1.to_csv("results_features_good/two_subtypes_freq.csv", index=False)
df1 = df1.sort_values(by=['features'])
df1.to_csv("results_features_good/two_subtypes_feat.csv", index=False)
print(df1.head())
# # print(df1.head())
# # #   continent  year  lifeExp
# # # 0      Asia  1952   28.801
# # # 1      Asia  1957   30.332
# # # 2      Asia  1962   31.997
# # # 3      Asia  1967   34.020
# # # 4      Asia  1972   36.088
# #
# # # ind   feature type acc freq
# #
# #
# # # pandas pivot
heatmap1_data = pd.pivot_table(df1, values='accuracy',
                     index=['features'],
                     columns='type')
sns.heatmap(heatmap1_data, cmap="YlGnBu")
plt.show()
