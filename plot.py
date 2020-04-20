import pickle
import matplotlib.pyplot as plt
import numpy as np

def boxplots_results(results, title=None):
    """
    Create a boxplot of the accuracy of the different feature
    selection methods.
    """
    sns.boxplot(x='accuracy',y='classifier',data=results)

    plt.gca().set_title('cross validation results')
    plt.gca().set_ylabel('classifier')
    plt.savefig('results.png')
    plt.show()

def feature_plot(features,feature_selectors,results):

    plt.figure()

    for selector in feature_selectors:

        std = []
        mean = []

        for i in range(len(features)):

            acc = results[i].loc[results[i]['feature selection'] == selector]['accuracy']

            std.append(np.std(acc))
            mean.append(np.mean(acc))

        plt.errorbar(features, mean, std,label=selector,marker='o',capsize=5)

    plt.legend()
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.savefig('featureplot.png',dpi=300)
    plt.show()

def make_plot(feature_selectors,test_list, pred_list):

    #Create confusion table based on the prediction values
    labels=["HER2+","HR+","Triple Neg"]

    for i in range(len(feature_selectors)):

        cm = confusion_matrix(test_list[i], pred_list[i],labels,normalize='true')            #This builds a confusion matrix by comparing diagnosis from the Test set (True diagnosis) against the predicted diagnosis (Y_pred)
        #print(cm)

        #Visualize the confusion table
        sns.set(font_scale=1.4) # for label size
        sns.heatmap(cm, annot=True, annot_kws={"size": 16},xticklabels=labels,yticklabels=labels,cmap="Blues")
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.show()
        plt.savefig("confusion{}.png".format(feature_selectors(i)))

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

    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)

    features = [10,20,40,50,60,70,80,90,100]
    feature_selectors = ["ReliefF","InfoGain","RFE"]

    feature_plot(features,feature_selectors,results)
