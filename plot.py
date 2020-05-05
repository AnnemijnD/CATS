import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def feature_plot(features,feature_selectors,results):

    plt.figure()

    for i,selector in enumerate(feature_selectors):

        std = []
        mean = []

        for ind,values in results[i].items():

            std.append(values[1])
            mean.append(values[0])

        plt.errorbar(features, mean, std,label=selector,marker='o',capsize=5)

    plt.legend()
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.savefig('featureplot.png',dpi=600)
    plt.show()

def make_plot(feature_selectors,pred_list,test_list):

    #Create confusion table based on the prediction values
    labels=["HER2+","HR+","Triple Neg"]

    for i in range(len(feature_selectors)):

        cm = confusion_matrix(test_list[i], pred_list[i],labels,normalize='true')            #This builds a confusion matrix by comparing diagnosis from the Test set (True diagnosis) against the predicted diagnosis (Y_pred)

        #Visualize the confusion table
        sns.set(font_scale=1.4) # for label size
        sns.heatmap(cm, annot=True, annot_kws={"size": 16},xticklabels=labels,yticklabels=labels,cmap="Blues")
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig("confusion_results.png",dpi=600)
        plt.show()

"""def plot_cm(name, model, xtrain, ytrain, xtest, ytest, ax=None, print_report=True):
    random.seed(0)
    np.random.seed(0)
    model.fit(xtrain, ytrain)
    fig = plot_confusion_matrix(model, xtest, ytest,
                      cmap=plt.cm.Blues, xticks_rotation='vertical',
                      normalize='pred', values_format='.2f', ax=ax)
    fig.ax_.set_title("confusion matrix: "+str(name))
    ypred = model.predict(xtest)
    plt.show()
    if print_report:
        print("===report:", str(name))
        print(classification_report(ytest, ypred))
"""

if __name__ == "__main__":

    with open('results.pkl', 'rb') as f1:
        prev_results = pickle.load(f1)

    with open('par_opt.pkl', 'rb') as f2:
        par_opt = pickle.load(f2)

    features = [10,20,30,40,50,60,70,80,90,100]
    feature_selectors = ["ReliefF","InfoGain","RFE"]

    best = []
    bestscore = 0
    best_ReliefF = []
    bestscore_ReliefF = 0
    best_IG = []
    bestscore_IG = 0
    best_RFE = []
    bestscore_RFE = 0

    for i in range(len(par_opt)):

        if par_opt[i]['selector'] == 'ReliefF':
            if par_opt[i]['mean'] > bestscore_ReliefF:
                bestscore_ReliefF = par_opt[i]['mean']
                best_ReliefF = par_opt[i]

        if par_opt[i]['selector'] == 'InfoGain':
            if par_opt[i]['mean'] > bestscore_IG:
                bestscore_IG = par_opt[i]['mean']
                best_IG = par_opt[i]

        if par_opt[i]['selector'] == 'RFE':
            if par_opt[i]['mean'] > bestscore_RFE:
                bestscore_RFE = par_opt[i]['mean']
                best_RFE = par_opt[i]

        if par_opt[i]['mean'] > bestscore:
            bestscore = par_opt[i]['mean']
            best = par_opt[i]

    print(best_ReliefF['selector'],
            best_ReliefF['features'],
            best_ReliefF['max_iter'],
            best_ReliefF['RELIEFF_K'],
            best_ReliefF['mean'],
            best_ReliefF['std'])

    print(best_IG['selector'],
            best_IG['features'],
            best_IG['max_iter'],
            best_IG['RELIEFF_K'],
            best_IG['mean'],
            best_IG['std'])

    print(best_RFE['selector'],
            best_RFE['features'],
            best_RFE['max_iter'],
            best_RFE['RELIEFF_K'],
            best_RFE['mean'],
            best_RFE['std'])

    ReliefF_results = {}
    IG_results = {}
    RFE_results = {}

    final_results = [ReliefF_results,IG_results,RFE_results]

    for Nfeatures_ind in range(len(features)):

        for ind,selector in enumerate(feature_selectors):

            for i in range(len(par_opt)):

                if par_opt[i]['features'] == Nfeatures_ind:

                    if par_opt[i]['selector'] == selector:

                        if ind == 0 and par_opt[i]['RELIEFF_K'] == 9 and par_opt[i]['max_iter'] == 800:

                            final_results[ind][Nfeatures_ind] = [par_opt[i]['mean'],par_opt[i]['std']]

                        elif par_opt[i]['RELIEFF_K'] == 7 and par_opt[i]['max_iter'] == 800:

                            final_results[ind][Nfeatures_ind] = [par_opt[i]['mean'],par_opt[i]['std']]

                        if par_opt[i]['selector'] == 'RFE' and par_opt[i]['features'] == 4:

                            pred_test_results = par_opt[i]['pred_test'][4]

    feature_plot(features,feature_selectors,final_results)

    make_plot(feature_selectors,pred_test_results[0],pred_test_results[1])
