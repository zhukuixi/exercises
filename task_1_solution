""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""

import nltk
import re
import os


#Define exceptions
class Task1Error(Exception): pass
class FileNoFoundError(Task1Error): pass

class dataPreprocess:
    """
    This class does data preprocess:
    1.lower case transformation
    2.text tokenize
    3.stop words removing
    """
    def __init__(self):
        """
        initialization
        self.wpt:The tokenizer we used to do tokenize
        self.stopword: The list of stopwords
        self.dic_stopword: A dictionary of stopwords for efficient stopwords removing
        """
        self.wpt=nltk.tokenize.WordPunctTokenizer()
        self.stopword_list=nltk.corpus.stopwords.words()
        self.dic_stopword={}
        for i in range(len(self.stopword_list)):
            self.dic_stopword[self.stopword_list[i]]=1
        self.document=""
        
    def FileInput(self,inputFile):
        """
        accept existed file
        @param inputFile: the address of input file
        @return: the addfreess of input file if it exists
        """
        #read the inputFile
        if not os.path.exists(inputFile):
            raise FileNoFoundError, "The input file %s does not exist!!"  % inputFile
        resultFile=""           
        resultFile=inputFile
        return resultFile
      
    def ExecutePreprocess(self,inputFile,sentenceLevel=False):
        """
        data preprocess
        @param inputFile: a string describing the address of inputfile
        @param sentenceLevel: indicator of whether we should do preprocess at sentence level
        @return: data after prepreocess
        """
        #turn the inputFile address into document
        inputFile=self.FileInput(inputFile)          
        deal_file=open(inputFile)
        deal_corpus=deal_file.read()
        self.document=deal_corpus
        if(sentenceLevel==False):
            toks=self.PreprocessWholeDocument(deal_corpus)           
        if(sentenceLevel==True):
            toks=self.PreprocessSentenceLevel(deal_corpus)        
        return toks
    
    def PreprocessWholeDocument(self,document):
        """
        Preprocess the data at document level
        @param document: the input document, which is a single string
        @return: the after preprocessed data, which is a single list
        """
        #tokenize
        toks = self.wpt.tokenize(document)
            
        #lower case transformation and only obtain words that are letters
        toks = [tok.lower() for tok in toks if tok.isalpha()]
           
        #get rid of stopwords efficiently
        toks = [tok for tok in toks if self.dic_stopword.has_key(tok)==False]
        return toks

        
    def PreprocessSentenceLevel(self,document):
        """
        Preprocess the data at sentence level
        @param document: the input document, which is a single string
        @return: the after preprocessed data, which is a list consisted of multiple lists corresponding to sentences
        """
        sentences=document.splitlines()
        
        #lower case transformation 
        sentences = [sent.lower() for sent in sentences]
            
        #tokenize
        toks = [self.wpt.tokenize(sent) for sent in sentences]
        return toks

class Task1:
    """
    The first method PopularTermAnalysis serves to find the most and least term in the document
    The second method GuitarTypeAnalysis serves to find different guitar types
    """
    def __int__(self):
        """
        initialization
        """
        self.mostPopularTerm=()
        self.leastPopularTerm=()       
    
    def PopularTermAnalysis(self,tokens):
        """
        @param tokens: tokens of a document
        @return
        """
        #get the frequency of each tokens
        term_fd=nltk.FreqDist(tokens)
        
        #get he most and least popular term based on the frequency count
        self.mostPopularTerm=(term_fd.items()[0][0],term_fd.items()[0][1])
        self.leastPopularTerm=(term_fd.items()[-1][0],term_fd.items()[-1][1])
        return [self.mostPopularTerm,self.leastPopularTerm]

    def GuitarTypeAnalysis(self,tokens):
        """
        @param tokens: tokens of a document
        @return: a list storing different guitar types
        Here we assume that the phrase describing guitar type should be a pair of words.
        The tag of the first word should not be "DT","IN" and 'VP', and the second word should be 'guitar', which is a noum ('NN').
        """
        #adding tag to each word
        tag_deal=[nltk.pos_tag(sent) for sent in tokens]
        guitarTypes=[]
        for index in range(len(tag_deal)):
            for word in tag_deal[index]:
                next_index=tag_deal[index].index(word)+1
                if next_index<len(tag_deal[index]):
                    if (word[1] not in ('DT','IN','VP')) and tag_deal[index][next_index]==('guitar', 'NN'):                    
                        guitarTypes.append(word[0])
                    else:
                        continue
        
        return sorted(set(guitarTypes))



if __name__ == '__main__':
    """
    Data preprocess
    """
    doPreprocess=True
    if(doPreprocess):
        print 'Preprocess Start!'
        preprocess=dataPreprocess()
        toks=preprocess.ExecutePreprocess("C:\Users\Kuixi\Desktop\NLP\data\deals_test.txt",sentenceLevel=False)
        toks_sen=preprocess.ExecutePreprocess("C:\Users\Kuixi\Desktop\NLP\data\deals_test.txt",sentenceLevel=True)
        print 'Preprocess Done!'

    """
    Use Task1 class to do
    1.Popular term analysis 
    2.Guitar type analysis
    """
    doTask1=True
    if(doTask1):
        print 'Task1 Start!'
        t1=Task1()
        
        #Popular term analysis
        t1.PopularTermAnalysis(toks)
        print t1.mostPopularTerm
        print t1.leastPopularTerm

        #Guitar type analysis
        guitarTypes=t1.GuitarTypeAnalysis(toks_sen)
        print guitarTypes
    
""" Report
1. Data preprocess
   For the frequency task, we should do analysis at the whole document level.
   For guitar type task, we should do analysis at sentence level.
   Thus, I set a parameter (sentenceLevel) in the ExecutePreprocess() function to generate after preprocessed data in different level.

2.Gutiar type task
   Without any prior knowlege of guitar type, the first thing come to mind is to do POS and parsing.
   I tried the dependecy parsing but the result is not promising. For example, for "online fingerstyle guitar lesson", the output of stanford parser is
   (NP (JJ online) (NN fingerstyle) (NN guitar) (NNS lesson))), where fingerstyle is pointing to lesson instead of guitar.
   Thus, I tried a simplier method mentioned above. The assumpiton is a bit naive but the result is pretty good. We correctly identify 9 kinds of guitar and errorly identy 2 kinds of guitar (
   1. learn guitar and online guitar)
"""
    

