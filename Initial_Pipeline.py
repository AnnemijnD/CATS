# Initial pipeline for Breast Cancer type prediction.
# This is just a preliminary example of how we could process the data and apply a CV method + a classifying method on our data
# and plot the results using a confusion table. I tried a randomized CV method with 25% and SVM as classifier,
# but we can change the methods easily. Note that there IS NO feature selection method at all, this needs to be implemented.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from ReliefF import ReliefF
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns
import plot
import itertools
import pickle

style.use('seaborn-whitegrid')
sns.set()

#This section can be pasted under any of the other subsections to generate a file containing the desired dataframe.
# It is just a way to visualize what each step does in a nice txt format instead of the console.
# Just paste it under the desired section and change "dftovisualize" to the df name you want to see
"""
open('checkpoint.txt', 'w').close()     # Deletes everything that was written in the file before
dftovisualize.to_csv(r'checkpoint.txt', header=None, index=None, sep=' ', mode='a')
"""

def process_data():

    """
    Processes patient data to a usable format.

    Returns:
        X (numpy array) : numpy dataframe with chromosomal data
        Y (numpy array) : numpy dataframe with diagnosis per patient
    """

    #Storing the data as two Dataframes
    callfile = "data/Train_call.txt"
    clinicalfile = "data/Train_clinical.txt"
    dfcall = pd.read_csv(callfile, delimiter="\t")
    dfclin = pd.read_csv(clinicalfile, delimiter="\t")

    #Tell number of samples and variables of the dataset
    print("Dataset dimensions : {}".format(dfcall.shape))

    #Check whether there is any "null" or "na" in the table (there is not) so no need to print it
    dfcall.isnull().sum()
    dfcall.isna().sum()

    #Rotate dfcall 90 degrees. This is needed because we want to visualize information per patient.
    #To do so, data must be rotated so each row is now a patient with a specific combination of 0,1,2 values per column
    # (chromosome) and a diagnosis as last column

    temp_df = dfcall.T                      #Transposes the dataframe
    rotated_df=temp_df[4::]                 #Removes the first 4 lines corresponding to the chromosomal locations and clone number (we do not need them for the moment)
    rotated_df = rotated_df.reset_index()   #Sets the new index based on the new number of rows (needed for adding the Diagnosis column afterwards)
    #print("rot", rotated_df)

    #Add the column of the diagnosis from dfclin at the end of the rotated dfcall. Now each patient (row) has a combination
    # of 0,1,2 values, that we have to link to a diagnosis. It can be found in dfclin dataframe.

    final_df=rotated_df.assign(Diagnosis=dfclin.Subgroup)   #Adds a column Diagnosis with the information of dfclin "Subgroup" column

    # Store separately the values and the diagnosis. This step is needed for the classifier,
    # we need to give separately the values from the diagnosis

    X = final_df.iloc[:,1:2835].values      #Store in X all the row data (without sample name or diagnosis). NOTICE that this takes ALL the features, usually we would apply a feature selection method
    Y = final_df.iloc[:, -1].values
          #Store in Y all the diagnosis ("Tripneg","HR+",...)
    return X, Y

def FS_ReliefF(X_train,Y_train,X_test):
    """
    Feature selection using ReliefF

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:

        X_fil: filtered dataframe
    """
    fs = ReliefF(n_neighbors=RELIEFF_K, n_features_to_keep=N_FEATURES)

    fs.fit(X_train,Y_train)

    X_train_fil = fs.transform(X_train)
    X_test_fil = fs.transform(X_test)

    return X_train_fil,X_test_fil

def FS_RFE(X_train, Y_train,X_test):
    """
    Feature selection using RFE-SVM

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:

        X_fil (numpy array): filtered dataframe
    """

    estimator = SVC(kernel="linear")
    selector = RFE(estimator, N_FEATURES, step=1)
    selector = selector.fit(X_train, Y_train)

    # construct mask
    mask = []
    for i in range(len(selector.support_)):
        if not selector.support_[i]:
            mask.append(i)

    X_train_fil = np.delete(X_train, mask, 1)
    X_test_fil = np.delete(X_test, mask, 1)

    return X_train_fil,X_test_fil

def FS_IG(X_train,Y_train,X_test):
    """
    Feature selection using FS_IG

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:
        X_fil: filtered dataframe
    """

    # gets the gains vector
    gain_vec = mutual_info_classif(X_train, Y_train, discrete_features=True)

    # gets the indices of columns that can be deleted from the dataset
    delete_ind = gain_vec.argsort()[::-1][N_FEATURES:]

    # deletes the features that can be deleted
    X_train_fil = np.delete(X_train, delete_ind, 1)
    X_test_fil = np.delete(X_test, delete_ind, 1)

    return X_train_fil,X_test_fil

def CV(X, Y):
    #
    #Divide the data into Training and Test set (25% of the data), selected randomly (for the moment).
    # This step needs to be adapted to the CV method we choose.
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    return X_train, X_test, Y_train, Y_test

def classify(X_train, X_test, Y_train, Y_test):
    #Apply Support Vector Machine Algorithm for classification using the train set. I chose this method as an example,
    # we need to apply the ones we chose in here.
    classifier = SVC(kernel = 'linear', random_state = 0)   #This builds the classifier using the SVC method
    classifier.fit(X_train, Y_train)                        #Using the training data (X for the 0,1,2 and Y for the diagnosis associated)

    #Predict the test set results using SVM model
    Y_pred = classifier.predict(X_test)                     #This predicts the diagnosis (Y_pred) of the test set data (X_test)

    score = accuracy_score(Y_test,Y_pred)

    return score,Y_pred

def summarize(scores):
    """

    returns the mean, the standard deviation, the min and the max
    of the cross validation accuracy.

    Args:
        scores (numpy array): accuracy scores of the cross validation

    Returns:
        summary (dict):
            mean: mean of scores
            std: standard deviation of scores
            min: minimum of scores
            max: maximum of scores
    """

    summary = {'mean':np.mean(scores),
                'std':np.std(scores),
                'max':max(scores),
                'min':min(scores)}

    return summary

def cross_validate(X,Y,Nsplits):

    entries = []
    ind = 0
    pred_list = []
    test_list = []

    for selector in feature_selectors:

        if selector != 'ReliefF' and RELIEFF_K > 7:
            continue

        print('\n starting with {}...\n'.format(selector))

        ind = 0

        this_pred_list = []
        this_test_list = []

        testing = []

        for iter in range(Niterations):

            validator = KFold(n_splits=Nsplits,shuffle=True)

            for train_index, test_index in validator.split(X):

                ind +=1

                print('starting with iteration {} of cross validation for {}'.format(ind,selector))

                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                if selector == 'ReliefF':
                    X_train_fil,X_test_fil = FS_ReliefF(X_train,Y_train,X_test)
                if selector == 'RFE':
                    X_train_fil,X_test_fil = FS_RFE(X_train,Y_train,X_test)
                if selector == 'InfoGain':
                    X_train_fil,X_test_fil = FS_IG(X_train,Y_train,X_test)

                score,Y_pred = classify(X_train_fil,X_test_fil,Y_train,Y_test)

                this_pred_list.append(Y_pred.tolist())
                this_test_list.append(Y_test.tolist())

                entries.append((selector,iter,score))

        this_pred_list = list(itertools.chain.from_iterable(this_pred_list))
        this_test_list = list(itertools.chain.from_iterable(this_test_list))
        pred_list.append(this_pred_list)
        test_list.append(this_test_list)

    return pd.DataFrame(entries, columns=['feature selection', 'iteration', 'accuracy']),pred_list,test_list

if __name__ == "__main__":

    features = [10,20,30,40,50,60,70,80,90,100]
    max_iter_list = [800,1000]
    RELIEFF_K = 10
    RELIEFF_K_list = [7,8,9]
    Niterations = 2
    Nsplits = 4 # for cross validation
    feature_selectors = ["ReliefF","InfoGain","RFE"]
    X, Y = process_data()

    par_opt = []
    par_opt_results = []

    for RELIEFF_K in RELIEFF_K_list:

        for max_iter in max_iter_list:

            classifier = SVC(kernel = 'linear', random_state = 0, max_iter=max_iter)

            results = []
            pred_test_results = []

            for i,N in enumerate(features):

                print("\n ----- Now calculating for {} features ------- \n".format(N))

                N_FEATURES = N

                # different train sets for the feature selection methods

                these_results,pred_list,test_list = cross_validate(X,Y,Nsplits)

                results.append(these_results)
                pred_test_results.append([pred_list,test_list])

            for selector in feature_selectors:

                for i in range(len(features)):

                    acc = results[i].loc[results[i]['feature selection'] == selector]['accuracy']
                    pred_test = pred_test_results[i]
                    std = np.std(acc)
                    mean = np.mean(acc)
                    par_opt.append({'selector':selector,
                                    'features':i,
                                    'max_iter':max_iter,
                                    'RELIEFF_K':RELIEFF_K,
                                    'mean':mean,
                                    'std':std,
                                    'results':acc,
                                    'pred_test':pred_test})

    with open('par_opt2.pkl', 'wb') as f1:
        pickle.dump(par_opt, f1)

    with open('results2.pkl', 'wb') as f2:
        pickle.dump(results, f2)

    #plot.feature_plot(features,feature_selectors,results)

"""Sources:
https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py"""
