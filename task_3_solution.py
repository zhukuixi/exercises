from __future__ import division
import task_1
import task_2
import nltk
import numpy
import sklearn.feature_extraction.text
import numpy as np
import operator
from gensim import corpora, models, similarities,matutils
from itertools import chain
from nltk.corpus import stopwords
from operator import itemgetter
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier



class Task3():
    """
    This class reads training and test data.
    """
    def __init__(self):
        """
        initialization
        """
        #the vectorizer, it could be "count' or 'tfidf'
        self.vectorizer = None
        
        #raw data and labels in different sets  (train/devtest/test/predict)
        self.X_train = None
        self.Y_train = None
        self.X_devtest=None
        self.Y_devtest=None
        self.X_predict = None
        self.Y_predict = None
        self.X_Test = None
        self.Y_Test = None
        
        #tokens of data             
        self.train_toks = None
        self.devtest_toks =None
        self.finaltest_toks=None
        self.predict_toks = None
        self.whole_toks= None
        
 
        #whole raw data
        self.whole_train= None
        
        #indicator of whether adding a new feature
        self.addNewFeature =False
      
        
    
    def setVectorizer(self,whole_deal_file,vectorizeStyle='tfidf'):
        '''
        Set the type of Vectorizer we use to transform the data
        It could be 'TfidfVectorizer' or 'CountVectorizer'
        '''
        stopwords_english=nltk.corpus.stopwords.words()
        
        corpus_deal=open(whole_deal_file)
        #whole_deal=corpus_deal.read()[:4000]
        whole_deal=corpus_deal.read()
        
        t2=task_2.Task2()
        whole_toks=t2.Stemmer(whole_deal)
        

        if vectorizeStyle=='tfidf':
            tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stopwords_english)
            self.vectorizer=tfidf_vectorizer.fit(whole_toks)
            
        elif vectorizeStyle=="count":
            count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=stopwords_english)
            self.vectorizer=count_vectorizer.fit(whole_toks)
    
    def readTrainData(self,good_deals_file, bad_deals_file):
        """
        read training data and transform it into a matrix
        1.adding new feature as a new column to the matrix
        2.adding the index of the deals as a new column to the matrix for the sake of raw text retrieval
        3.adding class labelas a new column to the matrix
        
        @param good_deals_file
        @param bad_deals_file
        @return
        """
        
        corpus_good=open(good_deals_file, 'r')
        corpus_bad=open(bad_deals_file, 'r')
        #count labels
        good_size=len(corpus_good.read().splitlines())
        bad_size=len(corpus_bad.read().splitlines())
     
        #combine documents
        whole_train=""
        for line in chain(open(good_deals_file, 'r'),open(bad_deals_file, 'r')):
            whole_train=whole_train+line
        self.whole_train=whole_train.split("\n")
        
        #Data Preprocess and vectorize
        t2=task_2.Task2()
        toks=t2.Stemmer(whole_train)
        development_toks=toks
        matrix=self.vectorizer.transform(toks) #development
        labels=np.array([1]*good_size + [0]*bad_size)
              
        #add new feature
        size_development=[]
        for line in development_toks:
            size_development.append(len(line.split()))  ##artificial new feature: count of words
            #size_development.append(len(line))  ##artificial new feature: length of deals
        matrix=np.column_stack((matrix.toarray(),size_development))

        ## attch the index to the development matrix to facilitate raw text retrieve
        indexMark=[]
        for index in range(matrix.shape[0]):
                indexMark.append(index)
        matrix=np.column_stack((matrix,indexMark))

      
        
        #attach labels
        matrix=np.column_stack((matrix,labels))

        #shuffle the matrix
   
        return matrix
    
    def matrixShuffle(self,matrix):
        '''
        Shuffle the row of matrix
        '''
        np.random.shuffle(matrix)
        return matrix
    
    def readPredictData(self,test_deals_file):
        '''
        read the test_deals.txt as predict data
        '''
        #read data
        test_file=open(test_deals_file)
        test_document=test_file.read()

        #Data Preprocess and vectorize
        t2=task_2.Task2()
        toks=t2.Stemmer(test_document)
        self.predict_toks=toks
        tfidf_matrix=self.vectorizer.transform(toks)
        
        self.X_predict = tfidf_matrix     
        self.Y_predict = None

        print self.X_predict
        
        ## attach the new feature to the predict matrix
        size_predict=[]
        for line in self.predict_toks:
            size_predict.append(len(line.split()))
        self.X_predict=np.column_stack((self.X_predict.toarray(),size_predict))

        print size_predict
        print self.X_predict
        print 'readPredictData Done!'
        return self.X_predict

    def data_distribution(self,matrix,predict_matrix,addNewFeature=False):
        '''
        This function serves to divide the whole data into 'training set', 'development test set' and 'test set'.
        Meanwhile, it will modify the data set given the value of addNewFeature (boolean value).
        If addNewFeature==Ture, we will include the new artificial feature in our dataset.
        @param matrix: the output of readTrainData, which is the matrix representing good_deals.txt and bad_deals.txt
        @param predict_matrix: the output of readPredictData, which is the matrix representing text_deals.txt
        @param addNewFeature : the indicator of whether or not adding new feature to the matrix
        '''
        self.addNewFeature=addNewFeature
        matrix_rowSize=matrix.shape[0]
        matrix_colSize=matrix.shape[1]
        #Assign data to training set
        XY_train = matrix[range(int(numpy.multiply(matrix_rowSize,0.8))),:]
        
        #Deciding whether or not we should include the new feature (for training set)
        self.Y_train = XY_train[:,-1]
        if addNewFeature:
            self.X_train=XY_train[:,range(matrix_colSize-2)]
        else:
            self.X_train=XY_train[:,range(matrix_colSize-3)]
        
        #Assign data to development test set
        XY_devtest= matrix[range(int(numpy.multiply(matrix_rowSize,0.8)),int(numpy.multiply(matrix_rowSize,0.9))),:]
        self.Y_devtest = XY_devtest[:,-1]
        
        #Decide whether or not we should include the new feature (for development test set)
        if addNewFeature:
            self.X_devtest=XY_devtest[:,range(matrix_colSize-1)]
        else:
            self.X_devtest=XY_devtest[:,range(matrix_colSize-1)]  #We don't delete the index feature here because we want to do raw data retrieval in error analysis
            
        #Assign data to the test set
        XY_Test= matrix[range(int(numpy.multiply(matrix_rowSize,0.9)),matrix_rowSize),:]
        self.Y_test = XY_Test[:,-1]

        #Decide whether or not we should include the new feature (for test set)
        if addNewFeature:
            self.X_test=XY_Test[:,range(matrix_colSize-2)]
        else:
            self.X_test=XY_Test[:,range(matrix_colSize-3)]

        #Decide whether or not we should include the new feature (for predict set)
        if addNewFeature:
            self.X_predict=predict_matrix
        else:
            self.X_predict=self.X_predict[:,range(predict_matrix.shape[1]-1)]

       
            
    def cv(self,estimator, X_train, Y_train, k_fold=5, scorelist=['accuracy', 'roc_auc']):
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
    
    
    def errorAnalysis(self,classifier):
        '''
        Conduct error analysis of the input classifier
        Here, we use the input classifier to predict the class label of data in THE development set
        This function will output those mis-classified deals and their class labels, which will help us tune the classifier and feature selection.
        @param classifier:the input classifier
        '''
        num_column=self.X_devtest.shape[1]
        num_row=self.X_devtest.shape[0]
        
        #predict the development test set
        if self.addNewFeature:
            guess=classifier.predict(self.X_devtest[:,range(num_column-1)])
        else:
            guess=classifier.predict(self.X_devtest[:,range(num_column-2)])
        index=self.X_devtest[:,-1]
        
        #error analysis
        errors=[]
        for row in range(num_row):
            if guess[row]!=self.Y_devtest[row]:             
                errors.append((self.Y_devtest[row],guess[row],self.whole_train[int(index[row])]))
        return errors
                
    def singlePerformanceEvaluation(self,clfList):
        '''
        Given the fixed test set, this functions serves to calculate the accuracy of the input classifers on the text set.
        @param clfList: the list of candidate classifier
        '''
        dic={}
        for clf,name in clfList:         
            pred=clf.predict(self.X_test)
            name
            acc=metrics.accuracy_score(self.Y_test, pred)
            dic[name]=acc
        return dic
    
    def allPerformanceEvaluation(self,development_matrix,predict_matrix,times=100):
        '''
        This functions serves to calculate the overall performance of classifiers. In this case, the accuracy of each classifier will be calculated given different test set and training
        set. Here we separate the data 100 times to alleviate the problem that the performance of a classifier greatly depends on the content of training set and test set.
        @param development_matrix: the matrix will be further separated as training set, development test set and test set
        @param predict_matrix:the data represents test_deals.txt
        @param times: the number of times we separate the data into different sets
        '''
        all_performance={"Naive Bayes":[],"SVM_rbf":[],"SVM_lin":[],"AdaBoost":[],"RandomForest":[]}
        for i in range(times):
            print "+++++++++++++++++++++++++++   "+str(i)+"      ++++++++++++++++"
            matrix=tfidf_clf.matrixShuffle(development_matrix) #matrix shuffle
            tfidf_clf.data_distribution(development_matrix,predict_matrix,True) #data separate
            nb_clf=tfidf_clf.doNBclassifier()
            svm_clf_rbf=tfidf_clf.doSVMclassifier("rbf")
            svm_clf_lin=tfidf_clf.doSVMclassifier("linear")
            ada_clf=tfidf_clf.doAdaBoostclassifier()
            rf_clf=tfidf_clf.doRandomForestclassifier()
            
            dic=tfidf_clf.singlePerformanceEvaluation([(nb_clf,"Naive Bayes"),[svm_clf_rbf,"SVM_rbf"],[svm_clf_lin,"SVM_lin"],[ada_clf,"AdaBoost"],[rf_clf,"RandomForest"]])
            for key in dic.keys():
                value=dic[key]
                all_performance[key].append(value)
        for key in all_performance.keys():
            accuracyList=all_performance[key]
            print key+" "+str(sum(accuracyList)/len(accuracyList))+" "+str(numpy.median(accuracyList))

    def ensemblePrediction(self,clf_list):
        '''
        A ensemble classifier to predict test_deals.txt
        Each classifier in clf_list is euqally weighted here.
        @param clf_list: the member of our ensemble classifier committee
        '''
        final_decision=[]
        for deal in self.X_predict:
            vote=[]
            for clf in clf_list:
                vote.append(clf.predict(deal))
            print vote
            #We assign different vote value with different marks
            #'Good*'/'Bad*' indicates a confidential classification result since all three classifiers agree that the deal is a good/bad one.
            if sum(vote)==3:
                final_decision.append("Good*")
            if sum(vote)==2:
                final_decision.append("Good")
            if sum(vote)==1:
                final_decision.append("Bad")
            if sum(vote)==0:
                final_decision.append("Bad*")  
         
                
        return final_decision
    
    def outputPredcitFile(self,predict_deals,predict_label):
        '''
        Output the predict result for test_deals.txt
        @predict_deals : the address of test_deals.txt
        @predict_label: the preidcted labels from the ensemble classifier
        '''
        predict_document=open(predict_deals)
        outputPredict_file=open(r"C:\Users\Kuixi\Desktop\NLP\data\test_deals[predicted].txt",'w')
        count_row=0
        for line in predict_document:
            outputPredict_file.write(line.rstrip()+"\t"+predict_label[count_row]+"\n")
            count_row=count_row+1

            
    def doNBclassifier(self,NBstyle="gaussian"):
        '''
        Naive Bayes classifier
        @param NBstyle: Deciding which NB classifier we should use. Here we have 'GaussianNB()' and 'MultinomialNB()'
        '''
        print 'NB Start!'
        if NBstyle=="gaussian":
            nb = GaussianNB()
        if NBstyle=="count":
            nb=MultinomialNB()
        
        if type(self.X_train) is numpy.ndarray:
            X_train=self.X_train
            X_test=self.X_test
        else:
            X_train=self.X_train.toarray()
            X_test=self.X_test.toarray()
            
        #use cross-validation to check the performance of the classifier
        nb_scores = self.cv(nb,X_train,self.Y_train)
        nb.fit(X_train, self.Y_train)

        #error analysis
        err=self.errorAnalysis(nb)
        print err
        print 'NB Done!'+'\n'
        return nb

        
    def doSVMclassifier(self,kernelStyle="rbf"):
        print 'SVM Start!'
        svm_clf=svm.SVC(kernel=kernelStyle,probability=True)
        if type(self.X_train) is numpy.ndarray:
            X_train=self.X_train
            X_test=self.X_test
        else:
            X_train=self.X_train.toarray()
            X_test=self.X_test.toarray()

        #use cross-validation to check the performance of the classifier
        svm_scores = self.cv(svm_clf,X_train,self.Y_train)
        svm_clf.fit(X_train, self.Y_train)

        #error analysis
        err=self.errorAnalysis(svm_clf)
        print err
        print 'SVM Done!'+'\n'
        return svm_clf

    def doAdaBoostclassifier(self):
        print 'Adaboost Start!'
        ada_clf=AdaBoostClassifier()
        if type(self.X_train) is numpy.ndarray:
            X_train=self.X_train
            X_test=self.X_test
        else:
            X_train=self.X_train.toarray()
            X_test=self.X_test.toarray()
            
        #use cross-validation to check the performance of the classifier
        ada_scores = self.cv(ada_clf,X_train,self.Y_train)
        ada_clf.fit(X_train, self.Y_train)

        #error analysis
        err=self.errorAnalysis(ada_clf)
        print err
        print 'Adaboost Done!'+'\n'
        return ada_clf

    def doRandomForestclassifier(self):
        print 'RandomForest Start!'
        rf_clf=RandomForestClassifier()
        if type(self.X_train) is numpy.ndarray:
            X_train=self.X_train
            X_test=self.X_test
        else:
            X_train=self.X_train.toarray()
            X_test=self.X_test.toarray()
            
        #use cross-validation to check the performance of the classifier
        rf_scores = self.cv(rf_clf,X_train,self.Y_train)
        rf_clf.fit(X_train, self.Y_train)

        #error analysis
        err=self.errorAnalysis(rf_clf)
        print err
        print 'RandomForest Done!'+'\n'
        return rf_clf


    
if __name__ == '__main__':
    #Read input data
    tfidf_clf = Task3()
    tfidf_clf.setVectorizer("C:\Users\Kuixi\Desktop\NLP\data\deals.txt")
    development_matrix=tfidf_clf.readTrainData(r"C:\Users\Kuixi\Desktop\NLP\data\good_deals.txt",r"C:\Users\Kuixi\Desktop\NLP\data\bad_deals.txt")
    predict_matrix=tfidf_clf.readPredictData(r"C:\Users\Kuixi\Desktop\NLP\data\test_deals.txt")
    matrix=tfidf_clf.matrixShuffle(development_matrix)

    #Data separate to training set/development test set /test set
    tfidf_clf.data_distribution(development_matrix,predict_matrix,False)  #We don't consider the new feature here
    tfidf_clf.doNBclassifier()
    tfidf_clf.doSVMclassifier("rbf")
    tfidf_clf.doSVMclassifier("linear")
    tfidf_clf.doAdaBoostclassifier()
    tfidf_clf.doRandomForestclassifier()


    print "############## after adding the new feature ########"
    tfidf_clf.data_distribution(development_matrix,predict_matrix,True) #We consider the new feature here
    nb_clf=tfidf_clf.doNBclassifier()
    svm_clf_rbf=tfidf_clf.doSVMclassifier("rbf")
    svm_clf_lin=tfidf_clf.doSVMclassifier("linear")
    ada_clf=tfidf_clf.doAdaBoostclassifier()
    rf_clf=tfidf_clf.doRandomForestclassifier()

    ##Now we decided to take the new feature into consideration and test the performance (accuracy) of our candidate classifiers
    tfidf_clf.allPerformanceEvaluation(development_matrix,predict_matrix)

    ## We select the top 3 classifiers as the component of our last ensemble classifier
    predict_label=tfidf_clf.ensemblePrediction([svm_clf_rbf,rf_clf,svm_clf_lin])

    #Final step, output the predict deals and labels.
    tfidf_clf.outputPredcitFile(r"C:\Users\Kuixi\Desktop\NLP\data\test_deals.txt",predict_label)
    
    


        
## [First Run]
## First, I seperate the data into test set,development test and test set. We use development test set for error analysis so we can improve our further feature selection.
## Second, I use test set to test different classifiers.
## Given the large number of features and limited number of samples, I dont want to focus on parameter tunning here because in this case it will definitely result in model
## overfitting. Instead, I attempt to put more weights on feature selection part.
## After the first run, the cross-validation result for these classifiers are not good. The error analysis give us some clues. Here it seems that deals ends with a URL tend
## te be a bad deal. However, I also found several good deals ends with a URL. Thus, more data is needed to decide whether we should take the appearance of URL at the end of
## deal as a new feature. Fortunatelly, there is a obvious difference between good deals and bad deals, which is the number of words. Thus, I added this feature in our data
## and see how it works in the second run.
'''
readPredictData Done!
NB Start!
# The cross-validation output of Naive Bayes Classifier
accuracy = 0.731 (0.102)
roc_auc = 0.816 (0.130)
'''
#The error analysis of Naive Bayes Classifier. In this case , (0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com') means that the correct label should be 0.0, but
#we falsely classified it as 1.0.
'''
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com'), (0.0, 1.0, 'Shop Swimsuits ForAll.com')]
NB Done!

SVM Start!
# The cross-validation output of SVM (Kernel=rbf) Classifier
accuracy = 0.558 (0.099)
roc_auc = 0.753 (0.150)

#The error analysis of SVM (Kernel=rbf) Classifier.
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com'), (0.0, 1.0, 'Shop Swimsuits ForAll.com')]
SVM Done!

SVM Start!
# The cross-validation output of SVM (Kernel=linear) Classifier
accuracy = 0.733 (0.096)
roc_auc = 0.717 (0.112)

#The error analysis of SVM (Kernel=linear) Classifier.
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com')]
SVM Done!

Adaboost Start!
# The cross-validation output of Adaboost Classifier
accuracy = 0.711 (0.096)
roc_auc = 0.809 (0.114)

#The error analysis of Adaboost Classifier
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com')]
Adaboost Done!

RandomForest Start!
# The cross-validation output of RandomForest Classifier
accuracy = 0.649 (0.121)
roc_auc = 0.817 (0.105)

# The error analysis of Randomforest Classifier
[(0.0, 1.0, 'Shop Swimsuits ForAll.com')]
RandomForest Done!
'''

## [Second Run]
## In the second run, the performance of all these classifiers imporve.
## Take a close look at the Naive Bayes classifier, it only gains little improvement in the roc_auc but get the same error analysis output as we have before considering the new feature.
## It is understandable since Naive Bayes assign equal weight to each feature and thus adding one informative feature won't improve the performance of the classifier too much.
## Unlike Naive Bayes Classifier, all the other classifier got huge improvement. For example, the SVM (kernel=rbf) perfectly predict all the deals in the development test set.
## Interestingly, the deal ''Cleveland Cavaliers Tickets Only at VividSeats.com'' again successfully escaped our classifiers, even though we added in a informative feature.
## Nevertheless, it is understandable since this deal got relatively large number of words so our new feature does not work well as before for this deal.
## Thus, we can consider new feature selection method. For exaple, we can consider n-gram or vectorize the text in character level. (I have not done these feature selection yet)

'''
############## after adding the new feature ########
NB Start!
# The cross-validation output of Naive Bayes Classifier
accuracy = 0.731 (0.102)
roc_auc = 0.853 (0.140)

#The error analysis of Naive Bayes Classifier.
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com'), (0.0, 1.0, 'Shop Swimsuits ForAll.com')]
NB Done!

SVM Start!
# The cross-validation output of SVM (Kernel=rbf) Classifier
accuracy = 0.813 (0.124)
roc_auc = 0.966 (0.059)

#The error analysis of SVM (Kernel=rbf) Classifier.
[]
SVM Done!

SVM Start!
# The cross-validation output of SVM (Kernel=linear) Classifier
accuracy = 0.836 (0.104)
roc_auc = 0.957 (0.067)

#The error analysis of SVM (Kernel=linear) Classifier
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com')]
SVM Done!

Adaboost Start!
# The cross-validation output of Adaboost Classifier
accuracy = 0.793 (0.087)
roc_auc = 0.907 (0.079)

#The error analysis of Adaboost Classifier
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com')]
Adaboost Done!

RandomForest Start!
# The cross-validation output of RandomForest Classifier
accuracy = 0.833 (0.051)
roc_auc = 0.898 (0.128)

#The error analysis of RandomForest Classifier
[(0.0, 1.0, 'Cleveland Cavaliers Tickets Only at VividSeats.com')]
RandomForest Done!
'''

##[Performance evaluation]
## Here we use the test set to test the performance of our classifier. (we used accuracy metric here)
## Given the very limited size of training set and test set, The trained classifier as well as the test set may not be representative.
## Thus, I decided to separate data into different training set and test set by doing shuffling. I did the separating 5000 times and record the average accuracy for each classifers.
## The result is showed as follows:

'''
The accuracy of the classifiers after 100 times iterations (separete data into different training set and test set)
SVM_rbf 0.861666666667 
RandomForest 0.845 
Naive Bayes 0.748333333333 
AdaBoost 0.791666666667
SVM_lin 0.838333333333
'''

## Based on accuracy, I decided to use RandomForest, SVM_rbf as well as SVM_lin as the component of our final ensemble classifier.
## For simpilicity, they are equally weight.

## [Future Work]
## I noticed that 52 out of 58 deals in test_deals.txt had been classifid as "Good*" or "Bad*", which means that all three classifier agree with each other on these deals.
## Thus, we can use semi-supervised method here by taking these confidential deals into our training set and re-train our classifiers.

'''
Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

[Report]
1. Divide the data into training set, development test set and test set;
2. Evalutation the performance of classifiers by using cross-validation in training set; Conduct error analyis on development test set;
3. Adding the count of word as a new feature to the data;
4. Shuffling the data and seperate the data into differnet training set, development test set and test set 100 times. Get the overall accuracy for each classifiers
5. Get the top 3 classifiers as the components of our final ensemble classifer.
6. Use the final classifier to predict the deals in test_deals.txt

 Q:- How general is your classifier?
 A: Given the very limited training set as well as the limited features we have in the training set, the trained classifier can hardly be a general one since it
    only represents its training data instead of the corpus of the deals. However, by adding a general feature (the count of word for deal), the classifier has
    better performance and became more general. Moreover, I used a ensemble classifier to collect vote from 3 classifiers. This ensemble method also makes the classifier
    more general. Finally, the proposed semi-supervised method, which means that taking the highly confidential deals in test_deals.txt into our training set and re-train the classifier
    could be a feasible method to make our classifier more robust and general. By doing these, we can keep predicting deals in deals.txt and increase the size of our training set to make
    the classifier more general.
    
 Q: - How did you test your classifier?
 A: In this task, the performance of the classifier greatly depends on the content of training set and test set. In other words, performance of the same classifer can vary greatly given
     different training set and test set. Thus I shuffled the data and seperated the data into differnet training set, development test set and test set 100 times. Finally, each classifier
     got 100 accuracy values. By calculating the average accuracy value for each classifiers, I can got a sense about the overall performance of each classifiers.
 '''
