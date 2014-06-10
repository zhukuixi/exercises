import numpy
import operator
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
from operator import itemgetter
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


class Task4:
    def readCSV(self,path_to_csv):
        '''
        Read the CSV frile
        @param path_to_csv: the address of csv file
        @return: the matrix form of the csv file
        '''
        matrix=numpy.loadtxt(open(path_to_csv,"rb"),delimiter=",",skiprows=1)
        return matrix

    def correlation(self,matrix):
        '''
        Caculate the pearson correlation coefficient for each pair of columns (feature) in the matrix
        @param :input matrix from CSV file
        @return : pearson correlation coefficient and the corresponding p-value
        '''
        dic_pvalue={}
        dic_pearson={}
        column=matrix.shape[1]
        for i in range(column-1):
            for j in xrange(i+1, column-1):
                r = pearsonr(matrix[:,i], matrix[:,j])
                dic_pvalue[(i,j)]=r[1]  #store the p-value for the pearson correlaiton coefficient for column i and j
                dic_pearson[(i,j)]=r[0] #store the pearson correlaiton coefficient for column i and j
                
        return (dic_pvalue,dic_pearson)
    
    
    def exploreRedundancy(self,correlation_result):
        '''
        These function serves to find out features which are significantly corrleated (p-value<0.05)
        @param correlation_result: the output of correlation(), which contains p-value as well as the pearson correlation coefficient
                
        '''
        dic_pvalue=correlation_result[0]
        dic_pearson=correlation_result[1]
        count=0
        answer=[]
        for key in dic_pvalue.keys():
            if dic_pvalue[key]<0.05:  #print out paris of feature whose p-value less then 0.05
                count=count+1
                print key,dic_pearson[key],dic_pvalue[key]
                answer.append([key,dic_pearson[key],dic_pvalue[key]])
        answer=sorted(answer, key=lambda x: x[2]) 
        print count
        #print out the Top 10 correlated pairs of features
        for ele in answer[:10]:
            print ele
            
        return answer

    def cv(self,estimator, X_train, Y_train, k_fold=5):
        '''
        Cross validation--considering accuracy only
        @param estimator: the classifier
        @param X_train: the training data
        @param Y_train: the class label of training data
        @param k_fold: the K fold cross-validation
        '''
        scoreArray = cross_val_score(estimator, X_train, Y_train, 'accuracy', k_fold)
        avg_accuracy = scoreArray.mean()
        return avg_accuracy
                
    def cv_multiscore(self,estimator, X_train, Y_train, k_fold=5, scorelist=['accuracy', 'recall','precision','roc_auc']):
        '''
        Cross validation
        @param estimator: the classifier
        @param X_train: the training data
        @param Y_train: the class label of training data
        @param k_fold: the K fold cross-validation
        @scorelist: metrics use to evaluate the performance of input classifier
        '''
        scores = {}
        for score in scorelist:
            scoreArray = cross_val_score(estimator, X_train, Y_train, score, k_fold)
            print("%s = %0.3f (%0.3f)" % (score, scoreArray.mean(), scoreArray.std()))
            scores[score] = scoreArray
        return scores
        
    def findBestFeatureForClf_PCA(self,clfList,X_train,Y_train):
        '''
        Given the candidate classifiers, this function serves to find out the optimal number of principle component for each of these classifiers
        @param clfList: the list of candidate classifiers
        @param X_train: training data set
        @param Y_train: the class label of training data set
        @return : return the optimal number of feature for each classifer, the name of each classifier and the accuracy of the classifier
        '''
        all_clf=[]
        for clf,name in clfList:  #go through each candidate classifer
            single_clf={}   #a dictionary records the accuracy of a single classifier given different number of features
            for featureCount in range(1,X_train.shape[1]+1): # go through different number of features
                print name,featureCount
                pca=PCA(n_components=featureCount)
                temp_X_train=pca.fit_transform(X_train)
                result=self.cv(clf,temp_X_train,Y_train) # the average accuracy after 5-fold cross validation
                single_clf[featureCount]=result
                
            single_clf = sorted(single_clf.iteritems(), key=operator.itemgetter(1),reverse=True)
            bestFeatureCount=single_clf[0][0] #get the best number of feature
            bestAccuracy=single_clf[0][1]     #get the corresponding best accuracy
            print name,bestFeatureCount,bestAccuracy
            all_clf.append([name,bestFeatureCount,bestAccuracy])

        all_clf=sorted(all_clf, key=lambda x: x[2],reverse=True)
        print all_clf
        return all_clf
        
    def generateCandidateClf(self):
        '''
        Generate a list of classifiers
        '''
        nb_clf = GaussianNB()
        svm_lin=svm.SVC(kernel='linear',probability=True)
        svm_rbf=svm.SVC(kernel='rbf',probability=True)
        ada_clf=AdaBoostClassifier()
        rf_clf=RandomForestClassifier()
        clfList=[(nb_clf,"Naive Bayes"),(svm_lin,"SVM Linear"),(svm_rbf,"SVM rbf"),(ada_clf,"Adaboost"),(rf_clf,"RandomForest")]
        #clfList=[(nb_clf,"Naive Bayes"),(svm_lin,"SVM Linear")]
        return clfList

    def fitBestFeature(self,all_clf,X_train,Y_train):
        '''
        Apply the optimal number of feature to the classifer with the highest accuracy
        @param all_clf:the output result of findBestFeatureForClf_PCA, which storing the name of classifer, the optimal number of feature and the accuracy of the classifier
        @param X_train: training data set
        @param Y_train: the class label of training data set
        '''
        best_clf_name=all_clf[0][0] #get the name of classifer with the highest accuracy
        best_feature_count=all_clf[0][1] #get the corresponding optimal number of feautre
        
        if best_clf_name=="SVM Linear":
            clf=svm.SVC(kernel='linear',probability=True)
        if best_clf_name=="Naive Bayes":
            clf=GaussianNB()
        if best_clf_name=="SVM rbf":
            clf=svm.SVC(kernel='rbf',probability=True)
        if best_clf_name=="Adaboost":
            clf=AdaBoostClassifier()
        if best_clf_name=="RandomForest":
            clf=RandomForestClassifier()

        pca=PCA(best_feature_count)
        temp_X=pca.fit_transform(X_train)
        clf.fit(temp_X,Y_train)
        return (clf,pca)
        
    def BestFeatureGeneralTest(self,clfList,X_train,Y_train,n_components):
        '''
        Apply the optimal number of feature to all classifier and use cross-validation to see how general the optimal features is to other classifiers
        @param clfList: the list of candidate classifiers
        @param X_train: training data set
        @param Y_train: the class label of training data set
        @n_components: the optimal number of feature
        '''
        
        all_clf={}
        for clf,name in clfList:
            single_clf={}         
            pca=PCA(n_components)
            temp_X_train=pca.fit_transform(X_train)
            result=self.cv_multiscore(clf,temp_X_train,Y_train)
            all_clf[name]=result

        for clf_name in all_clf.keys():
            score=all_clf[clf_name]
            print clf_name
            for metric in score.keys():
                print metric,score[metric].mean()
                
        return all_clf


   

if __name__ == "__main__":
    t4=Task4()
    matrix=t4.readCSV('C:\Users\Kuixi\Desktop\NLP\data\coupon_clickstream.csv')
    X_train=matrix[:,range(matrix.shape[1]-1)]
    Y_train=matrix[:,matrix.shape[1]-1]
    redundacy_explore=True
    featureSearch=True

    if redundacy_explore:
        ##redundacy explore
        print "redundacy explore start!"
        correlation_result=t4.correlation(matrix)
        answer=t4.exploreRedundancy(correlation_result)
        print "redundacy explore done!"
        
    if featureSearch:
        ## Find the optimal number of feature for each classifier
        print "Optimal feature searching start!"
        clfList=t4.generateCandidateClf()
        all_clf=t4.findBestFeatureForClf_PCA(clfList,X_train,Y_train)
        result=t4.fitBestFeature(all_clf,X_train,Y_train)
        print "Optimal feature searching done!"

        #get the best classifier and its reduced feature st
        clf=result[0]
        pca=result[1]
        #rank the features from most important to least.
        print pca.explained_variance_ratio_
        ## Test the generality of the best feature we found
        t4.BestFeatureGeneralTest(clfList, X_train,Y_train,pca.n_components)
    
     



        
# [Redundacy exploration]
# output format [column A, column B], pearson correlation coefficient, p-value]
# From here, we can see that there are lots of feature significantly correlated with each other. Particularlly, there are 5 pairs of feature which are identical to each other.
# Thus, we can answer the three questions. Features are redundant and some of them are correlated. Thus, it is necessary to apply dimension reduction.
'''
[(21, 24), 1.0, 0.0]  
[(13, 28), 1.0, 0.0]
[(7, 30), 1.0, 0.0]
[(35, 40), 1.0, 0.0]
[(33, 46), 1.0, 0.0]
[(30, 31), -0.68947345396711734, 5.8677668603387063e-142]
[(7, 31), -0.68947345396711734, 5.8677668603387063e-142]
[(42, 47), -0.68282129305681327, 3.2443864964221531e-138]
[(9, 41), 0.67950128467584148, 2.1972575064059478e-136]
[(35, 43), -0.64229712972428976, 2.0512521905250093e-117]
'''

#[Find the optimal number of feature for each classifier]
# I use PCA here for dimension reduction. In order to determine the optimal number of principle component to maximize the accuracy, for each candidate classifier, I run 5 fold
# cross-validation for each possible number ofprinciple component to get the accuracy. Thus, for each classifier, the number of features starts from 1 to 50.
#The following output format is ['name of classifier', optimal number of feature, accuracy]
## Finally, we choose SVM (kernel=linear) with 32 principle component. The corresponding eigen value of each principle component stands for the importance of the feature.
## Thus, we use 'pca.explained_variance_ratio_' to get the rank of the features from most important to least

'''
[['SVM rbf', 50, 0.87799999999999989],
['Adaboost', 15, 0.91999999999999993],
['Naive Bayes', 14, 0.93200000000000005],
['RandomForest', 21, 0.93999999999999984],
['SVM Linear', 32, 0.94600000000000006]]
]

'''
#[Best Feature General Test]
#Except SVM rbf,which will suffer decrease in accuracy, the best feature fits other classifiers well and thus it is general.
'''
SVM Linear
recall 0.958099009901
roc_auc 0.980939647965
precision 0.935689274017
accuracy 0.946
RandomForest
recall 0.882217821782
roc_auc 0.974049405941
precision 0.946085910653
accuracy 0.921
Naive Bayes
recall 0.922237623762
roc_auc 0.979619667967
precision 0.926203959169
accuracy 0.924
SVM rbf
recall 0.98
roc_auc 0.965739171917
precision 0.706357041359
accuracy 0.785
Adaboost
recall 0.904099009901
roc_auc 0.969279633963
precision 0.920712116162
accuracy 0.913
'''
