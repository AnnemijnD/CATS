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

if __name__ == "__main__":

    with open('results_nested.pkl', 'rb') as f:
        results = pickle.load(f)

    features = [10,20,30,40,50,60,70,80,90,100]
    feature_selectors = ["ReliefF", "InfoGain", "RFE"]
    Nsplits_out = 5

    best_results = {}
    best_scores = np.zeros((len(feature_selectors),len(features)))

    for i,selector in enumerate(feature_selectors):

        best_results[selector] = {}

        for j,Nfeatures in enumerate(features):

            best_score = 0

            for k in range(Nsplits_out):
                if results[selector][Nfeatures][k+1]['score'] > best_score:

                    best_score = results[selector][Nfeatures][k+1]['score']
                    best_scores[i,j] = best_score
                    best_results[selector][Nfeatures] = results[selector][Nfeatures][k+1]

    print('\n ------------- BEST SCORES ---------------- \n')
    for selector in feature_selectors:

        print('--- {} --- \n'.format(selector))

        for Nfeatures in features:
            print('for {} with {} features, the best accuracy is {}'.format(selector,Nfeatures,best_results[selector][Nfeatures]['score']))
            print('the corresponding parameters are: {}'.format(best_results[selector][Nfeatures]['params']))
            print('\n')

    plt.figure()
    for i,selector in enumerate(feature_selectors):
        plt.plot(features,best_scores[i,:],label=selector)
    plt.legend()
    plt.xlabel('N features')
    plt.ylabel('Best accuracy score')
    plt.show()
