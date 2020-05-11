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
# dictionary: {Feature:[accuracies]}
# Total accuracy
FEAT_ACC = {}

# accuracy of TN
FEAT_ACCTN = {}

# accuracy HR+
FEAT_ACCHR = {}

# accuracy HER2+
FEAT_ACCHER2 = {}

# accuracy of the features
FREQ_FEATURES = {}
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
    print(final_df.head())
    # Store separately the values and the diagnosis. This step is needed for the classifier,
    # we need to give separately the values from the diagnosis
    final_df = final_df[final_df.Diagnosis != "HER2+"]
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
    # global CHOSEN_FEAUTURES
    fs = ReliefF(n_neighbors=RELIEFF_K, n_features_to_keep=N_FEATURES)

    fs.fit(X_train,Y_train)
    chosen_features = fs.top_features[:N_FEATURES]
    for el in chosen_features:
        CHOSEN_FEAUTURES.append(el)

    X_train_fil = fs.transform(X_train)
    X_test_fil = fs.transform(X_test)

    return X_train_fil,X_test_fil

def FS_RFE(X_train, Y_train,X_test,classifier):
    """
    Feature selection using RFE-SVM

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:

        X_fil (numpy array): filtered dataframe
    """

    selector = RFE(classifier,N_FEATURES, step=RFE_STEP)
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
    gain_vec = mutual_info_classif(X_train, Y_train,n_neighbors=IG_NEIGHBOURS,discrete_features=True)

    # gets the indices of columns that can be deleted from the dataset
    delete_ind = gain_vec.argsort()[::-1][N_FEATURES:]
    for i in range(len(gain_vec)):
        if i not in delete_ind:
            CHOSEN_FEAUTURES.append(i)
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

def cross_validate(X,Y,Nsplits,classifier):
    global CHOSEN_FEAUTURES

    entries = []
    ind = 0
    pred_list = []
    test_list = []

    for selector in feature_selectors:

        # if selector != 'ReliefF' and RELIEFF_K > 7:
        #     continue

        print('\n starting with {}...\n'.format(selector))

        ind = 0

        this_pred_list = []
        this_test_list = []

        testing = []
        for iter in range(Niterations):

            validator = KFold(n_splits=4,shuffle=True)

            for train_index, test_index in validator.split(X):

                ind +=1

                print('starting with iteration {} of cross validation for {}'.format(ind,selector))

                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                if selector == 'ReliefF':
                    X_train_fil,X_test_fil = FS_ReliefF(X_train,Y_train,X_test)
                if selector == 'RFE':
                    X_train_fil,X_test_fil = FS_RFE(X_train,Y_train,X_test,classifier)
                if selector == 'InfoGain':
                    X_train_fil,X_test_fil = FS_IG(X_train,Y_train,X_test)

                score,Y_pred = classify(X_train_fil,X_test_fil,Y_train,Y_test)

                this_pred_list.append(Y_pred.tolist())
                this_test_list.append(Y_test.tolist())

                entries.append((selector,iter,score))

                # calculate the accuracy of each subtype
                # types = ["Triple Neg", "HR+", "HER2+"]
                types = ["Triple Neg", "HR+"]
                typeaccs = []
                for type in types:
                    correct = 0
                    incorrect = 0

                    for y in range(len(Y_pred)):
                        if Y_test[y] == type:
                            if Y_pred[y] == Y_test[y]:
                                correct +=1
                            else:
                                incorrect +=1
                    try:
                        acc = correct / (correct + incorrect)
                    except:
                        acc = np.nan
                    typeaccs.append(acc)

                # update the dictionaries
                update_FSstats(score, typeaccs, types)

                # delete the list with chosen_features so it can be used next iteration
                del CHOSEN_FEAUTURES[:]

        this_pred_list = list(itertools.chain.from_iterable(this_pred_list))
        this_test_list = list(itertools.chain.from_iterable(this_test_list))
        pred_list.append(this_pred_list)
        test_list.append(this_test_list)

        # save the dictionaries
        save_features(selector)
    return pd.DataFrame(entries, columns=['feature selection', 'iteration', 'accuracy']),pred_list,test_list

def update_FSstats(score, typeaccs,types):
    """
    Updates the directionaries that keep up the accuracies of the features that were selected
    """
    global CHOSEN_FEAUTURES

    for feature in CHOSEN_FEAUTURES:
        if feature in FREQ_FEATURES:
            FREQ_FEATURES[feature] += 1
        else:
            FREQ_FEATURES[feature] = 1

        if feature in FEAT_ACC:
            FEAT_ACC[feature].append(score)
            FEAT_ACCTN[feature].append(typeaccs[0])
            FEAT_ACCHR[feature].append(typeaccs[1])
            # FEAT_ACCHER2[feature].append(typeaccs[2])
        else:
            FEAT_ACC[feature] = [score]
            FEAT_ACCTN[feature]= [typeaccs[0]]
            FEAT_ACCHR[feature] = [typeaccs[1]]
            # FEAT_ACCHER2[feature] = [typeaccs[2]]

def save_features(selector):
    """
    Save the features with the accuracies, take the mean per feature
    """
    features = []
    accs = []
    freqs = []
    accsTN = []
    accsHR = []
    # accsHER = []

    # append the lists per feature
    for feature in FREQ_FEATURES.keys():
        features.append(feature)
        accs.append(np.nanmean(FEAT_ACC[feature]))

        freqs.append(FREQ_FEATURES[feature])
        accsTN.append(np.nanmean(FEAT_ACCTN[feature]))
        accsHR.append(np.nanmean(FEAT_ACCHR[feature]))
        # accsHER.append(np.nanmean(FEAT_ACCHER2[feature]))

    # accstypes = [accsTN, accsHR, accsHER]
    accstypes = [accsTN, accsHR]
    # types = ["TN", "HR+", "HER2+"]
    types=["TN", "HR+"]

    # make a dict so it can be saved to csv
    dict2 = {"features":[], "type":[], "accuracy":[], "freqs":[]}
    for feature in range(len(features)):
        for i in range(len(accstypes)):
            dict2["features"].append(features[feature])
            dict2["type"].append(types[i])
            dict2["accuracy"].append(accstypes[i][feature])
            dict2["freqs"].append(freqs[feature])


    df = pd.DataFrame(data=dict2)
    df = df.sort_values(by=['freqs'])
    df.to_csv(f'results_features/paramopt/heatmap_{selector}_K={K}_MAX_ITER={MAX_ITER}_N={N_FEATURES}_2types.csv', index=False)
    # df = df.sort_values(by=['features'])
    # df.to_csv(f'results_features/heatmap_{selector}_feat.csv', index=False)

if __name__ == "__main__":
    # SVC optimization
    # degrees = [0, 1, 2, 3, 4, 5, 6]
    degrees = [1]
    # degrees = [0]
    # cs = [0.1, 1, 10, 100, 1000]
    cs = [0.1]
    # max_iter_list = [600,700,800]
    max_iter_list = [900]

    # number of features
    # features = [10,20,30,40,50,60,70,80,90,100]
    features = [50]
    # feature selector optimization
    # RELIEFF_K_list = [5,6,7,8,9,10]
    RELIEFF_K_list = [9]
    # RFE_step_list = [1,2,3]
    RFE_step_list = [3]
    # IG_neighbours_list = [2,3,4,5,6,7]
    IG_neighbours_list = [4]

    Niterations = 10

    # Nsplits_list = [3,4,5]
    Nsplits_list = [3]

    # feature_selectors = ["ReliefF"]#, "InfoGain", "RFE"]
    feature_selectors = ["RFE"]
    X, Y = process_data()

    par_opt = []
    par_opt_results = []

    for Nsplits in Nsplits_list:

        for IG_NEIGHBOURS in IG_neighbours_list:

            for RFE_STEP in RFE_step_list:

                for RELIEFF_K in RELIEFF_K_list:

                    for max_iter in max_iter_list:

                        for c in cs:

                            for degree in degrees:

                                classifier = SVC(kernel='linear',C=c,degree=degree,max_iter=max_iter)

                                results = []
                                pred_test_results = []

                                for i,N in enumerate(features):

                                    print("\n ----- Now calculating for {} features ------- \n".format(N))

                                    N_FEATURES = N
                                    MAX_ITER = max_iter
                                    K = RELIEFF_K

                                    # different train sets for the feature selection methods

                                    these_results,pred_list,test_list = cross_validate(X,Y,Nsplits,classifier)

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
                                                        'cs':c,
                                                        'degrees':degree,
                                                        'RELIEFF_K':RELIEFF_K,
                                                        'RFE_STEP':RFE_STEP,
                                                        'IG_NEIGHBOURS':IG_NEIGHBOURS,
                                                        'mean':mean,
                                                        'std':std,
                                                        'results':acc,
                                                        'Nsplits':Nsplits,
                                                        'pred_test':pred_test
                                                        })

    with open('par_opt2.pkl', 'wb') as f1:
        pickle.dump(par_opt, f1)

    with open('results2.pkl', 'wb') as f2:
        pickle.dump(results, f2)

    #plot.feature_plot(features,feature_selectors,results)

"""Sources:
https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py"""
