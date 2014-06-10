import nltk
import task_1
import pprint
import sklearn.feature_extraction.text
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import operator
from gensim import corpora, models, similarities
from itertools import chain
from nltk.corpus import stopwords
from operator import itemgetter


class Task2:
    
    def Stemmer(self,toks):
        '''
        Stem the document
        @param toks:input raw document, which is a string
        @return tokens after stemming process
        '''
        stemmer=nltk.stem.PorterStemmer()
        dp=task_1.dataPreprocess()
        toks=dp.PreprocessSentenceLevel(toks)
        nonInformative_words=[]

        for i in range(len(toks)):
            sentence=[]
            for word in toks[i]:
                if word in nonInformative_words:
                    continue
                if word.isalpha():
                    sentence.append(stemmer.stem(word))
            toks[i]=" ".join(sentence)
        return toks
    
    def DocumentVectorize(self,toks,lines=4000):
        '''
        Stem and transform the document into tfidf_matrix
        @param toks:input raw document, which is a string
        @return tfidf matrix and the vocabulary of the document
        '''
        #To avoid ValueError, we just analyze the first 4000 deals here
        deal_corpus=toks[:lines]
        
        #Tfidf vectorization
        stopwords_english = nltk.corpus.stopwords.words('english')
        tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stopwords_english)
        tfidf_matrix=tfidf_vectorizer.fit_transform(deal_corpus)
        return (tfidf_matrix,tfidf_vectorizer.vocabulary_)
    
    def KmeansCluster(self,matrix,n_clusters=10):
        km = sklearn.cluster.KMeans(n_clusters)
        km.fit(matrix.toarray())
        return km  

    def GroupAnalysis(self,matrix,voca,labels):
        '''
        This function serves to get the top tf-idf word for each group after the K-means clustering
        From here, we can see whether the K value we choose for clustering is meaningful
        By using this function, we can tune the K value of K-means clustering by checking coherence
        of the top 10 tf-idf words of each group

        @param matrix:the tf-idf matrix of the whole corpus
        @voca: the vocabulary(stemmed) of the whole corpus
        @labels: group label for each deal
        '''
        #data initialization
        labels=list(labels)
        array=matrix.toarray()
        unique_label=list(set(labels))
        numberOfGroup=len(unique_label)
        #group_document: a list of array storing partial tf_idf matrix for each group
        group_document=[np.array([[0.0,]*array.shape[1]]),]*numberOfGroup
        
        #group_dictionary: a list of dictionary representing each group. In each dictionary, the key is
        #word and the value is the sum of tf-idf of that word within the group
        group_dictionary=[dict() for x in range(numberOfGroup)]
        
        sorted_voca = sorted(voca.iteritems(), key=operator.itemgetter(1))
        Top10Words=[[],]*numberOfGroup

        #Given the group labels, we separate the original tf-idf matrix into different groups
        group_document=self.spliceMatrix(array,group_document,labels)
        #Within each group, we calcualte the sum of tf-idf value of each word
        group_dictionary=self.sumColumnForEachGroup(group_dictionary,group_document,numberOfGroup,sorted_voca)

        #For each group, get the Top 10 words with the highest summed up tf-idf value
        for group in range(numberOfGroup):
            Top10Words[group]=self.getTop10Words(group_dictionary[group])            
        return Top10Words
    
    def sumColumnForEachGroup(self,group_dictionary,group_document,numberOfGroup,sorted_voca):
        '''
        Sum up the tf-idf value of each word for each group
        For each group, we use a dicionary to store the {"word":"sum of its tf-idf value in thie group"}
        Praticularry, we set a {"GROUPSIZE":the size of the group} key-value pair for each dicitonary

        @param group_document: a list of array storing partial tf_idf matrix for each group
        @numberofGroup: the number of group.It equals to the K value in K-means clustering
        @sorted_voca: the sorted vocabulary dictionary, where the key is word and the value is the number
        of column in the tf_idf matrix representing this word.
        @return:return the gourp_dictionary
        
        '''
        for group in range(numberOfGroup):
            colSum=group_document[group].sum(axis=0)
            for index  in range(len(sorted_voca)):
                key=str(sorted_voca[index][0])
                value=colSum[index]
                group_dictionary[group][key]=value
            group_dictionary[group]["GROUPSIZE"]=len(group_document[group])
        return group_dictionary
    
    def spliceMatrix(self,array,group_document,labels):
        '''
        Seperate the whole matrix into different parts and assign them to group_document.
        So we can analyze the submatrixs of the original tf_idf matrix corresponding to each group
        respectively.
        
        @param array: the original tf_idf matrix
        @param group_document: a list of array storing partial tf_idf matrix for each group
        @param labels:group labels for each deal
        @return: the group_document
        '''
        for row in range(len(array)):
            if labels[row]==-1:
                continue
            else:
                old=group_document[int(labels[row])]
                new=array[row]
                new.shape=(1,array.shape[1])
                group_document[int(labels[row])]=np.concatenate((old,new),axis=0)
        return group_document
        
    def getTop10Words(self,inputDictionary):
        sorted_dic = sorted(inputDictionary.iteritems(), key=operator.itemgetter(1),reverse=True)
        answer=[]
        answer.append(sorted_dic[0][1])                                   
        for i in range(1,11):
            answer.append(sorted_dic[i][0])
        return answer

    def LdaAnalysis(self,inputFile,n_topics=15):
        '''
        This function achieves the 2nd goal of our task--Topic finding.
        The default number of topic is 15
        '''
        t1_dp=task_1.dataPreprocess()
        dic_stopword=t1_dp.dic_stopword
        #stem , tokenizing and stopwords, non-letter words removing
        toks=t2.Stemmer(inputFile)
        texts=[[word for word in nltk.tokenize.wordpunct_tokenize(line) if dic_stopword.has_key(word)==False and word.isalpha()] for line in toks]
        texts=texts[:4000]
        #dicitonary and corpus_tfidf generating
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        #specificy the number of topic and model fitting        
        n_topics = 15
        lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)

        for i in range(0, n_topics):
         temp = lda.show_topic(i, 10)
         terms = []
         for term in temp:
             terms.append(term[1])
         print "Top 10 terms for topic #" + str(i) + ": "+ ", ".join(terms)
         
        print 
        print 'Which LDA topic maximally describes a document?\n'
        print 'Original document: ' + toks[1]
        print 'Preprocessed document: ' + str(texts[1])
        print 'Matrix Market format: ' + str(corpus[1])
        print 'Topic probability mixture: ' + str(lda[corpus[1]])
        print 'Maximally probable topic: topic #' + str(max(lda[corpus[1]],key=itemgetter(1))[0])


    def ClusteringAnalysis(self,inputFile,n_clusters=15):
        '''
        This function achieve the first goal of out task--Clustering Analysis.
        The default number of cluster is 15
        '''
        toks=self.Stemmer(inputFile)
        tfidf_result=self.DocumentVectorize(toks,4000)
        tfidf_matrix=tfidf_result[0]
        tfidf_voca=tfidf_result[1]
        n_clusters=15
        #K means clustering
        km=t2.KmeansCluster(tfidf_matrix,n_clusters)
        labels=km.labels_
        #Clustering result analysis
        gd=t2.GroupAnalysis(tfidf_matrix,tfidf_voca,labels)
        return gd



if __name__ == '__main__':
    t2=Task2()
    deal_file=open("C:\Users\Kuixi\Desktop\NLP\data\deals.txt")
    deal_corpus=deal_file.read()
    doClusterAnalysis=False
    doTopicAnalysis=True
    
    if doClusterAnalysis:
        ca=t2.ClusteringAnalysis(deal_corpus,15)
    if doTopicAnalysis:
        ta=t2.LdaAnalysis(deal_corpus,15)

    
    
    
'''
The output of ClusteringAnalysis: For each row, the first element is the number of members of that group,
the following ten words are the words with the highest summed up tf-idf value within the group.

[2374, 'shop', 'product', 'gift', 'ivl', 'ticket', 'get', 'best', 'price', 'sale', 'find']
[261, 'free', 'ship', 'order', 'get', 'day', 'shop', 'gift', 'code', 'plu', 'ani']
[241, 'link', 'page', 'thi', 'direct', 'land', 'product', 'homepag', 'visitor', 'cvsphoto', 'home']
[203, 'save', 'samsung', 'onli', 'camera', 'wifi', 'tv', 'smarttv', 'led', 'digit', 'big']
[140, 'new', 'york', 'arriv', 'style', 'wweshop', 'line', 'avail', 'shop', 'product', 'game']
[118, 'deal', 'bookingbuddi', 'cheap', 'rock', 'great', 'flight', 'vacat', 'smart', 'offici', 'merchandis']
[113, 'offer', 'order', 'code', 'expir', 'coupon', 'ani', 'onli', 'use', 'enter', 'get']
[105, 'hotel', 'deal', 'book', 'buddi', 'save', 'chocolat', 'top', 'rate', 'orlando', 'lo']
[92, 'use', 'code', 'promo', 'coupon', 'save', 'devic', 'http', 'order', 'domain', 'entir']
[83, 'coupon', 'oblig', 'daili', 'busi', 'balm', 'help', 'code', 'insideup', 'compar', 'owner']
[74, 'text', 'link', 'taser', 'iclipart', 'gener', 'cart', 'dayspr', 'christian', 'vehicl', 'hotel']
[72, 'sep', 'valid', 'saturday', 'begin', 'sunday', 'remov', 'run', 'pleas', 'promot', 'thi']
[68, 'discount', 'shop', 'fabric', 'onlinefabricstor', 'net', 'great', 'travel', 'retail', 'offic', 'never']
[50, 'onlin', 'learn', 'guitar', 'lesson', 'artistwork', 'drum', 'jazz', 'mandolin', 'dobro', 'percuss']
[21, 'singl', 'brows', 'meet', 'faith', 'parent', 'christian', 'usedcar', 'text', 'view', 'photo']

Comment:
After trying different K value for Kmeans clustering, K=15 seems a meaningful value.
By having a closer look to the group analysis result and revisit the raw data by searching top 10 words of
each group, we can conclude that we have the following group across all the deals:

1. General product selling 
2. Deals emphasize 'Plus Free shipping'
3. Links difrect users to other webpages
4. Samsung digital products deals
5. Deals from New York and deals includes words "new arrivals"  (We should try to capture New York as a single word)
6. Deals from BookingBuddy.com
7. Online coupon deals
8. Hotel booking, traving related deals
9. Promotional sells
10. H-Balm deals and InsideUp helps small business owners get free no obligation  (It's a mixture and makes no much senses)
11. Text link to iCLIPART.com
12. Information describe when does the validity of coupon begins and ends. In most case, it contains Septemer.
13. Discount deals and fabric product deals
14. Online musical instrument learning lessons
15. Singles meeting deals

'''

'''
The output of Topic Analysis:

Top 10 terms for topic #0: hotel, style, save, vega, clearanc, price, samsung, deal, shop, low
Top 10 terms for topic #1: san, product, hard, diet, check, francisco, lumen, lenovo, new, way
Top 10 terms for topic #2: brows, new, wifi, ani, camera, date, school, free, tool, shop
Top 10 terms for topic #3: cheap, tv, flight, bookingbuddi, onli, gener, sun, free, digit, cold
Top 10 terms for topic #4: specif, link, cart, extra, direct, x, page, year, rental, top
Top 10 terms for topic #5: contact, lens, jewelri, america, save, silver, monster, north, gold, rebat
Top 10 terms for topic #6: directli, samsung, item, onli, led, save, goe, shop, link, discount
Top 10 terms for topic #7: valid, sep, pleas, promot, detail, code, free, check, remov, order
Top 10 terms for topic #8: coupon, code, expir, white, team, samsung, save, camera, order, short
Top 10 terms for topic #9: buddi, hotel, book, deal, newegg, code, use, printer, coupon, costum
Top 10 terms for topic #10: smart, save, deal, visitor, homepag, vacat, link, descript, page, photo
Top 10 terms for topic #11: textlink, page, free, cart, link, direct, enjoy, save, chang, make
Top 10 terms for topic #12: singl, free, ship, gift, map, order, link, text, meet, card
Top 10 terms for topic #13: link, canada, entir, text, free, extrem, small, user, certif, vacat
Top 10 terms for topic #14: norton, premier, link, onlin, avail, free, ship, order, deal, deep

Which LDA topic maximally describes a document?

Original document: Online country guitar lesson.
Preprocessed document: ['onlin', 'countri', 'guitar', 'lesson']
Matrix Market format: [(1, 1), (2, 1), (3, 1), (4, 1)]
Topic probability mixture: [(0, 0.01333333537398571), (1, 0.013333337992429511), (2, 0.01333333887464829), (3, 0.013333333492931847), (4, 0.013333347117108735), (5, 0.013333341487594278), (6, 0.013333336574880816), (7, 0.01333334151170063), (8, 0.01333333998617503), (9, 0.013333333405625768), (10, 0.013333336565392647), (11, 0.013333336854712257), (12, 0.013333335184359474), (13, 0.013333345824605361), (14, 0.81333325975384962)]
Maximally probable topic: topic #14


    
'''
