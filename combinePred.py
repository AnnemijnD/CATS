import pickle
import numpy as np
from sklearn.metrics import accuracy_score

def substitute_HER(Y_pred_HERremoved,Y_pred_HERpresent):

    for i in range(len(Y_pred_HERpresent)):
        if Y_pred_HERpresent[i] == 'HER2+':
            Y_pred_HERremoved[i] = 'HER2+'

    return Y_pred_HERremoved

if __name__ == "__main__":

    with open('datasets/results_HERremoved.pkl', 'rb') as f1:
        results_HERremoved = pickle.load(f1)

    with open('datasets/results_HERpresent.pkl', 'rb') as f2:
        results_HERpresent = pickle.load(f2)

    features = [10,20,30,40,50,60,70,80,90,100]
    feature_selectors = ["ReliefF", "InfoGain", "RFE"]
    Nsplits_out = 5
    best_score = 0

    improvement = 0

    for selector in feature_selectors:
        for Nfeatures in features:
            for ind_out in range(Nsplits_out):

                    Y_test = results_HERremoved[selector][Nfeatures][ind_out+1]['Y_test']

                    Y_pred_HERremoved = results_HERremoved[selector][Nfeatures][ind_out+1]['Y_pred']
                    score_HERremoved = accuracy_score(Y_test,Y_pred_HERremoved)
                    print('\n removed: \n')
                    print(Y_pred_HERremoved)

                    Y_pred_HERpresent = results_HERpresent[selector][Nfeatures][ind_out+1]['Y_pred']
                    score_HERpresent = accuracy_score(Y_test,Y_pred_HERpresent)
                    print('\n present: \n')
                    print(Y_pred_HERpresent)

                    Y_pred_improved = substitute_HER(Y_pred_HERremoved,Y_pred_HERpresent)
                    score_improved = accuracy_score(Y_test,Y_pred_improved)
                    print('\n improved: \n')
                    print(Y_pred_improved)

                    print('\n ----- removed, selector {}; Nfeatures {}; ind_out {}------ \n'.format(selector,Nfeatures,ind_out))
                    print(score_HERremoved)
                    print('\n ----- present, selector {}; Nfeatures {}; ind_out {}------ \n'.format(selector,Nfeatures,ind_out))
                    print(score_HERpresent)
                    print('\n ----- improved, selector {}; Nfeatures {}; ind_out {}------ \n'.format(selector,Nfeatures,ind_out))
                    print(score_improved)

                    if score_improved != score_HERpresent:

                        improvement += score_improved - score_HERpresent

    print('improvement = ', improvement/(len(feature_selectors)*len(features)*Nsplits_out))
