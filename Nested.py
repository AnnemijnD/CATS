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
from mpl_toolkits.mplot3d import Axes3D

style.use('seaborn-whitegrid')
sns.set()

global CHOSEN_FEAUTURES
CHOSEN_FEAUTURES = []
FEAT_ACC = {}
FEAT_ACCTN = {}
FEAT_ACCHR = {}
FEAT_ACCHER2 = {}
FREQ_FEATURES = {}
df_heatmap = {"feature":[], "accs":[]}
K = 0
MAX_ITER = 0
N_FEATURES = 0
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

def FS_ReliefF(X_train,Y_train,X_test,ReliefF_K):
    """
    Feature selection using ReliefF

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:

        X_fil: filtered dataframe
    """

    # global CHOSEN_FEAUTURES
    fs = ReliefF(n_neighbors=ReliefF_K, n_features_to_keep=N_FEATURES)

    fs.fit(X_train,Y_train)
    chosen_features = fs.top_features[:N_FEATURES]
    for el in chosen_features:
        CHOSEN_FEAUTURES.append(el)

    X_train_fil = fs.transform(X_train)
    X_test_fil = fs.transform(X_test)

    return X_train_fil,X_test_fil

def FS_RFE(X_train,Y_train,X_test,RFE_step,classifier):
    """
    Feature selection using RFE-SVM

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:

        X_fil (numpy array): filtered dataframe
    """

    selector = RFE(classifier,N_FEATURES,step=RFE_step)
    selector = selector.fit(X_train, Y_train)

    # construct mask
    mask = []
    for i in range(len(selector.support_)):
        if not selector.support_[i]:
            mask.append(i)
        else:
            CHOSEN_FEAUTURES.append(i)


    X_train_fil = np.delete(X_train, mask, 1)
    X_test_fil = np.delete(X_test, mask, 1)

    return X_train_fil,X_test_fil

def FS_IG(X_train,Y_train,X_test,IG_neighbours):
    """
    Feature selection using FS_IG

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:
        X_fil: filtered dataframe
    """

    # gets the gains vector
    gain_vec = mutual_info_classif(X_train, Y_train, discrete_features=True,n_neighbors=IG_neighbours)

    # gets the indices of columns that can be deleted from the dataset
    delete_ind = gain_vec.argsort()[::-1][N_FEATURES:]
    for i in range(len(gain_vec)):
        if i not in delete_ind:
            CHOSEN_FEAUTURES.append(i)
    # deletes the features that can be deleted
    X_train_fil = np.delete(X_train, delete_ind, 1)
    X_test_fil = np.delete(X_test, delete_ind, 1)

    return X_train_fil,X_test_fil

def classify(X_train, X_test, Y_train, Y_test,classifier):
    #Apply Support Vector Machine Algorithm for classification using the train set. I chose this method as an example,
    # we need to apply the ones we chose in here.
    #classifier = SVC(kernel = 'linear', random_state = 0)  !!! oh my god...
    classifier.fit(X_train, Y_train)                        #Using the training data (X for the 0,1,2 and Y for the diagnosis associated)

    #Predict the test set results using SVM model
    Y_pred = classifier.predict(X_test)                     #This predicts the diagnosis (Y_pred) of the test set data (X_test)

    score = accuracy_score(Y_test,Y_pred)

    return score,Y_pred

def check_accuracy(highest_acc_params,X_train_out,Y_train_out,X_test_out,Y_test_out):

    degree = highest_acc_params['inner_results']['degree']
    c = highest_acc_params['inner_results']['c']
    max_iter = highest_acc_params['inner_results']['max_iter']

    classifier = SVC(kernel='linear',C=c,degree=degree,max_iter=max_iter)

    if selector == 'ReliefF':

        ReliefF_K = highest_acc_params['inner_results']['ReliefF_K']
        X_train_out_fil,X_test_out_fil = FS_ReliefF(X_train_out,Y_train_out,X_test_out,ReliefF_K)
        score,Y_pred = classify(X_train_out_fil,X_test_out_fil,Y_train_out,Y_test_out,classifier)

    if selector == 'RFE':

        RFE_step = highest_acc_params['inner_results']['RFE_step']
        X_train_out_fil,X_test_out_fil = FS_ReliefF(X_train_out,Y_train_out,X_test_out,RFE_step)
        score,Y_pred = classify(X_train_out_fil,X_test_out_fil,Y_train_out,Y_test_out,classifier)

    if selector == 'InfoGain':

        IG_neighbours = highest_acc_params['inner_results']['IG_neighbours']
        X_train_out_fil,X_test_out_fil = FS_ReliefF(X_train_out,Y_train_out,X_test_out,IG_neighbours)
        score,Y_pred = classify(X_train_out_fil,X_test_out_fil,Y_train_out,Y_test_out,classifier)

    outer_accuracy_results = {'score':score,'params':highest_acc_params,'Y_test':Y_test_out,'Y_pred':Y_pred}

    return outer_accuracy_results

def average_accuracy(inner_results,Nsplits_in):

    avg_accs = []
    highest_acc = 0
    highest_acc_params = {}

    for i in range(len(inner_results[1])):

        avg_acc = 0

        for ind_in,inner_result in inner_results.items():

            avg_acc += inner_results[ind_in][i]['score']

        avg_acc = avg_acc/Nsplits_in
        avg_accs.append({'avg_acc':avg_acc,'inner_result':inner_results[1][i]})

        if avg_acc > highest_acc:
            highest_acc = avg_acc
            highest_acc_params = inner_results[1][i]

    highest_acc_params = {'avg_acc':avg_acc,'inner_results':highest_acc_params}

    return highest_acc_params

def parameter_optimization(X_train_in,Y_train_in,X_test_in,Y_test_in,selector,ind_in,inner_results):

    degrees = [3]
    cs = [1]
    max_iter_list = [800]

    # feature selector optimization
    RELIEFF_K_list = [7,8,9]
    RFE_step_list = [1,2]
    IG_neighbours_list = [2,3,4]

    Niterations = 1

    Nsplits_list = [4]

    for degree in degrees:

        for c in cs:

            for max_iter in max_iter_list:

                classifier = SVC(kernel='linear',C=c,degree=degree,max_iter=max_iter)

                if selector == 'ReliefF':

                    for ReliefF_K in RELIEFF_K_list:

                        X_train_fil,X_test_fil = FS_ReliefF(X_train_in,Y_train_in,X_test_in,ReliefF_K)
                        score,Y_pred = classify(X_train_fil,X_test_fil,Y_train_in,Y_test_in,classifier)

                        inner_results[ind_in].append({'selector':selector,'Nfeatures':N_FEATURES,'score':score, 'degree':degree, 'c':c, 'max_iter':max_iter, 'ReliefF_K':ReliefF_K})

                if selector == 'RFE':

                    for RFE_step in RFE_step_list:

                        X_train_fil,X_test_fil = FS_RFE(X_train_in,Y_train_in,X_test_in,RFE_step,classifier)
                        score,Y_pred = classify(X_train_fil,X_test_fil,Y_train_in,Y_test_in,classifier)

                        inner_results[ind_in].append({'selector':selector,'Nfeatures':N_FEATURES,'score':score, 'degree':degree, 'c':c, 'max_iter':max_iter, 'RFE_step':RFE_step})

                if selector == 'InfoGain':

                    for IG_neighbours in IG_neighbours_list:

                        X_train_fil,X_test_fil = FS_IG(X_train_in,Y_train_in,X_test_in,IG_neighbours)
                        score,Y_pred = classify(X_train_fil,X_test_fil,Y_train_in,Y_test_in,classifier)

                        inner_results[ind_in].append({'selector':selector,'Nfeatures':N_FEATURES,'score':score, 'degree':degree, 'c':c, 'max_iter':max_iter, 'IG_neighbours':IG_neighbours})

def nested_cross_validate(X,Y,Nsplits_out,Nsplits_in,selector,results):

    validator_out = KFold(Nsplits_out)

    ind_out = 0

    for train_index_out,test_index_out in validator_out.split(X):

        ind_out += 1

        print('\n starting with outer CV {} for {}...\n'.format(ind_out,selector))

        X_train_out, X_test_out = X[train_index_out], X[test_index_out]
        Y_train_out, Y_test_out = Y[train_index_out], Y[test_index_out]

        validator_in = KFold(Nsplits_in)

        ind_in = 0

        inner_results = {}

        for train_index_in,test_index_in in validator_in.split(X_train_out):

            ind_in += 1

            inner_results[ind_in] = [] # this one should have the same parameters at every index of the lists beloning to every ind_in

            print('starting with inner iteration {} of outer CV {} for {}'.format(ind_in,ind_out,selector))

            X_train_in, X_test_in = X_train_out[train_index_in], X_train_out[test_index_in]
            Y_train_in, Y_test_in = Y_train_out[train_index_in], Y_train_out[test_index_in]

            parameter_optimization(X_train_in,Y_train_in,X_test_in,Y_test_in,selector,ind_in,inner_results) #adds to inner_results

            # the same features need to be selected now for the outer test set.

        highest_acc_params = average_accuracy(inner_results,Nsplits_in)
        outer_accuracy_results = check_accuracy(highest_acc_params,X_train_out,Y_train_out,X_test_out,Y_test_out)

        results[selector][N_FEATURES][ind_out] = outer_accuracy_results

if __name__ == "__main__":

    # number of features
    features = [10,20,30,40,50,60,70,80,90,100]
    feature_selectors = ["RFE"] #["ReliefF", "InfoGain"]#, "RFE"]
    Nsplits_out = 5
    Nsplits_in = 4

    results = {}

    X, Y = process_data()

    for selector in feature_selectors:

        results[selector] = {}

        for N_FEATURES in features:

            results[selector][N_FEATURES] = {}

            print("\n ----- Now calculating for {} features for {} ------- \n".format(N_FEATURES,selector))

            nested_cross_validate(X,Y,Nsplits_out,Nsplits_in,selector,results) # adds to results

    for i in range(Nsplits_out):
        print('\n',results["ReliefF"][10][i+1]['score'])
        print(results["ReliefF"][10][i+1]['params'])

    with open('results_nested.pkl', 'wb') as f:
        pickle.dump(results, f)

    #plot.feature_plot(features,feature_selectors,results)

"""Sources:
https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py"""
