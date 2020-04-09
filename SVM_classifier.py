import pandas as pd
import numpy as np
#import cvxopt

#From the feature selection algorithm a couple of features are selected to be used in the classification algorithm, therefore:
#Read list with feature names/postitions
#For now lets use some random features..
callfile = "data/Train_call.txt"
clinicalfile = "data/Train_clinical.txt"
dfcall = pd.read_csv(callfile, delimiter="\t")
dfclin = pd.read_csv(clinicalfile, delimiter="\t")

#Select first 10 features because we don't have a feature selection method yet
dfcall = dfcall[0:10]
dfcall = dfcall.T
#Seperate gene info
dfgeneinfo = dfcall[0:3]
dfcall = dfcall[4:]
#dfcall = dfcall.T

##combine df call with dfclin to get the class label per sample
#dfclin = dfclin.rename(index = lambda s: dfclin[1,s])
#classlabels = dfclin["Subgroup"]
#arrays = dfclin["Samples"]
#dfclinnew = classlabels.append(arrays)
#dfcall = dfcall.append(dfclin.T)


print(dfgeneinfo)
print(dfcall)
print(dfclin)


#Do SVM
