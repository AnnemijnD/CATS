import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEAT_ACC = {}
FREQ_FEATURES = {}

def update_FSstats(accuracy,CHOSEN_FEAUTURES):

    for feature in CHOSEN_FEAUTURES:
        if feature in FEAT_ACC:
            FEAT_ACC[feature].append(accuracy)
        else:
            FEAT_ACC[feature] = [accuracy]

        if feature in FREQ_FEATURES:
            FREQ_FEATURES[feature] += 1
        else:
            FREQ_FEATURES[feature] = 1

def plot_features():

    #print("FEATACC", FEAT_ACC)
    #print("FREQ_FEATURES", FREQ_FEATURES)

    features = []
    accs = []
    freqs = []
    for feature in FREQ_FEATURES.keys():
        features.append(feature)
        accs.append(np.mean(FEAT_ACC[feature]))
        freqs.append(FREQ_FEATURES[feature])

    dict = {"features":features, "accs":accs, "freqs":freqs}
    df = pd.DataFrame(data=dict)
    # df.to_csv('out.csv', index=False)
    # plt.scatter(features, accs)
    # for i, txt in enumerate(features):
    #     # plt.annotate(txt, (freqs[i], accs[i]))

    # plt.show()

    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(features, accs, zs=freqs)
    ax.set_xlabel('Ca_cyt',fontsize=20,labelpad=15)
    ax.set_ylabel('Ca_ER', fontsize=20,labelpad=15)
    ax.set_zlabel('Ca_m',  fontsize=20,labelpad=15, rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = plt.gca()
    # ax.set_facecolor('#e9e9e9')
    plt.grid(color='white')
    # plt.show()
    # data_url = 'http://bit.ly/2cLzoxH'
    # gapminder = pd.read_csv(data_url)
    # print(gapminder.head(3))
    # df1 = gapminder[['continent', 'year','lifeExp']]
    # print(df1.head())
    # #   continent  year  lifeExp
    # # 0      Asia  1952   28.801
    # # 1      Asia  1957   30.332
    # # 2      Asia  1962   31.997
    # # 3      Asia  1967   34.020
    # # 4      Asia  1972   36.088
    #
    # # ind   feature type acc freq
    #
    #
    # # pandas pivot
    # heatmap1_data = pd.pivot_table(df1, values='lifeExp',
    #                      index=['continent'],
    #                      columns='year')
    # sns.heatmap(heatmap1_data, cmap="YlGnBu")
    # plt.show()
