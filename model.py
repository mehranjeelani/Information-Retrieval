from nltk.tokenize import RegexpTokenizer
import string as STR
from collections import Counter
import math
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import re
import nltk
nltk.download('punkt')
from sklearn.svm import LinearSVC
import itertools
import numpy as np
from scipy import stats

def retrieval_tfidf():
    score = {}
    global magnitude_doc
    magnitude_doc = {}
    for query in tqdm(queries.keys()):
        term_freq_query = queries[query]
        tfidf_query = {}
#       Calculating tfidf for query terms
        for term in term_freq_query.keys():
            tfidf_query[term] = term_freq_query[term]*idf.get(term,0)
        temp = {}
        magnitude_query = math.sqrt(sum([math.pow(tfidf_query[term],2) for term in tfidf_query.keys() ])) 
        for doc in corpus.keys():
            similarity_dot_product = 0
#           Calculating corresponding tfidf of the query terms in the document  
            for term in tfidf_query.keys():
                tfidf_term_doc = term_frequency.get(doc,{}).get(term,0)*idf.get(term,0)
                similarity_dot_product += tfidf_term_doc * tfidf_query[term]
            if doc not in magnitude_doc.keys():
                magnitude = 0
                for term in term_frequency[doc].keys():
                    magnitude += math.pow(term_frequency[doc][term]*idf[term],2)
                magnitude_doc[doc] = math.sqrt(magnitude)
           
            temp[doc] = similarity_dot_product/(magnitude_doc[doc]*magnitude_query)
        score[query] = temp
    return score

def precision(score):
    precision_50 = {}
    for query in tqdm(score.keys()):
        num_relevant = 0
        for doc,_ in score[query]:
#       if any of the regex pattern for the query answer matches against the doc, it is relevant
            if any (re.search(regex,corpus_raw[doc]) for regex in query_answer[query]): 
                num_relevant+=1
        #print('number of relevant docs for query {} are {}'.format(query,num_relevant))
        precision_50[query] = num_relevant/len(score[query])
    return precision_50

def retrieval_BM25():
    score = {}
    for query in tqdm(queries.keys()):
        temp = {}
        for doc,_ in score_1000[query]:
          
          temp[doc] = 0
          for term in queries[query].keys():
              numerator_constant = PARAM_K1 + 1
              denominator_constant = PARAM_K1 * (1 - PARAM_B + PARAM_B * len(corpus[doc]) / avg_dl)
              temp[doc] +=  idf_bm25.get(term,0)*term_frequency_unnormalized.get(doc,{}).get(term,0)*numerator_constant/(term_frequency_unnormalized.get(doc,{}).get(term,0)+denominator_constant)
        score[query] = temp
    return score

def retrieval_BM25_sentence(sentence_raw,sentence_tokenized,avg_sl):
  score = {}
  for query in tqdm(queries.keys()):
   
    temp = {}
    for index  in sentence_tokenized[query].keys():
        sentence = sentence_tokenized[query][index]
        term_frequency_unnormalized_sentence = Counter(sentence)
        temp[index] = 0
        for term in queries[query].keys():
            numerator_constant = PARAM_K1 + 1
            denominator_constant = PARAM_K1 * (1 - PARAM_B + PARAM_B * len(sentence) / avg_sl)
            temp[index] +=  idf_bm25.get(term,0)*term_frequency_unnormalized_sentence.get(term,0)*numerator_constant/(term_frequency_unnormalized_sentence.get(term,0)+denominator_constant)
    score[query] = temp
  return score

def retrieval_SVM():
  score_svm_50 = {}
  for query in score_1000.keys():
      X = [] # test data matrix
      temp =[]
      index_doc = {} # Used for mapping indices in data matrix to document idf. This will be later used for sorting.
#      Extracting features from the documents
      for doc_id,_ in score_1000[query]:
          feauture_tfidf = 0
          feature_tf = 0
          feature_idf = 0
          for term in queries[query].keys():
              feauture_tfidf+=  term_frequency_unnormalized.get(doc_id,{}).get(term,0)/len(corpus[doc_id])*idf.get(term,0)
              feature_tf += term_frequency_unnormalized.get(doc_id,{}).get(term,0)/len(corpus[doc_id]) #len(queries[query].keys())
              feature_idf += idf.get(term,math.log(Total_docs))/len(queries[query].keys())
          feature_dl = len(corpus[doc_id])/avg_dl
          feature_BM25 =  score_bm25[query][doc_id]
        
          X.append([feature_dl,feature_tf,feauture_tfidf,feature_BM25,feature_idf])
          index_doc[len(X)-1] = doc_id
      result = clf.decision_function(X)
      result_indices = np.argsort(-1*result)[:50] # Sorting in decreasing order of scores
      for index in result_indices:
          temp.append((index_doc[index],result[index])) 
      score_svm_50[query] = temp # This contains top 50 doc ids along with scores for each query 
  return score_svm_50

def retrieval_SVM_sentence(sentence_raw,sentence_tokenized,avg_sl,score_sentence_BM25):
  score_svm_50 = {}
  for query in tqdm(queries.keys()):
      X = []
      temp =[]
      index_sentence = {}
      for sentence_id in sentence_tokenized[query].keys():
          sentence = sentence_tokenized[query][sentence_id]
          term_frequency_unnormalised_sentence = Counter(sentence)
         
          #var = Counter(sentence)
          
          #term_frequency_sentence = {key:var[key]/var.most_common(1)[0][1] for key in var}
          
          feauture_tfidf = 0
          feature_tf = 0
          feature_idf = 0
#         Extracting features from the sentences
          for term in queries[query].keys():
              #if len(sentence) == 0: print('error: ',term_frequency_sentence)
              feauture_tfidf+=   term_frequency_unnormalised_sentence.get(term,0)*idf.get(term,0)/len(sentence)
              feature_tf += term_frequency_unnormalised_sentence.get(term,0)/len(sentence) 
              feature_idf += idf.get(term,math.log(Total_docs))/len(queries[query].keys())
          feature_sl = len(sentence)/avg_sl
          feature_BM25 = score_sentence_BM25[query][sentence_id]
        
          X.append([feature_sl,feature_tf,feauture_tfidf,feature_BM25, feature_idf])
          index_sentence[len(X)-1] = sentence_id
      result = clf.decision_function(X)
      result_indices = np.argsort(-1*result)[:50]
      for index in result_indices:
          temp.append((index_sentence[index],result[index])) 
      score_svm_50[query] = temp
  return score_svm_50

def MRR(score_sentence_50):
  mrr = 0.0
  rank_list = {}
  for query in tqdm(queries.keys()):
    
    
    for tuples,rank in zip(score_sentence_50[query],list(range(1,len(score_sentence_50[query])+1))):
      id = tuples[0]
      
      if any (re.search(regex,sentence_raw[query][id]) for regex in query_answer[query]):
        mrr += 1.0/rank
        rank_list[query] = rank
        #print('\nRank of sentence for query {} is {}.'.format(query,rank))
        break
    if not query in rank_list.keys():
      rank_list[query] = None

  return float(mrr/len(list(queries.keys()))),rank_list

# Transforming the dataset into pairwise comparison for training ranking svm  
def transform(X,y):
    X_new = []
    y_new = []
    y = np.asarray(y)
    #index_comb = {}
    
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
        # skip if same ranking or if they belong to different query 
          continue
        X_new.append(X[i] - X[j])
        #index_comb[len(X_new)-1] = (i,j)
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
      # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
            #index_comb[len(X_new)-1] = (j,i)
    return np.asarray(X_new), np.asarray(y_new).ravel()#, index_comb

def get_sentence(score):
    sentence_raw= {} # This will be used for comparing the sentence against regex pattern for relevance matching and extracting sentences
    sentence_tokenized = {} # This will be used for extracting BM25, tfidf etc
    for query in tqdm(score.keys()):
      temp = []
      sentence_raw[query] = {}
#     Splitting the top 50 docs for each query into individual sentences and adding sentence id to each sentence
      for doc,_ in score[query]:
        temp += sent_tokenize(corpus_raw[doc].strip())
      for id,sentence in zip(list(range(1,len(temp)+1)),temp):
        sentence_raw[query][id] = sentence
#     Tokenizing the sentences 
      sentence_tokenized[query] = {}
      for id in sentence_raw[query].keys():
        sentence = sentence_raw[query][id]
        temp =  sentence.lower().translate(table)
        temp = tokenizer.tokenize(temp)
        if len(temp)!=0 : sentence_tokenized[query][id] = temp
        
    #Calculation of average sentence length:
    total_sentence_length = 0
    num_of_sentences = 0
    for query in tqdm(sentence_tokenized.keys()):
      for id in sentence_tokenized[query].keys():
        total_sentence_length+=len(sentence_tokenized[query][id])
      num_of_sentences+=len(sentence_tokenized[query].keys())
    return sentence_raw,sentence_tokenized,total_sentence_length/(num_of_sentences*1.0)

with open('trec_documents.xml', 'r') as f:  # Reading file
    xml = f.read()
xml = '<ROOT>' + xml + '</ROOT>'
root = BeautifulSoup(xml, 'lxml-xml')
corpus = {}

for doc in tqdm(root.find_all('DOC')):
    if not doc.find('DOCNO').text.strip().startswith("LA"):
        corpus[doc.find('DOCNO').text.strip()] = doc.find('TEXT').text.strip()
    else:
        text = ""
        for child in  doc.find('TEXT').findChildren("P" , recursive=False):
          
            string = child.text.strip()
            text = ' '.join([text, string])

        corpus[doc.find('DOCNO').text.strip()] = text.strip()
# Dictionary corpus_raw contains raw document(without any pre-processing) as value of corpus_raw[doc_id]. This will be used for 
# pattern matching to find relevance later on.
corpus_raw = corpus.copy()

# Pre-processing the corpus. We are removing punctuations, performing tokenization and lowering the case. No stemming or lemmatization 
# has been performed
tokenizer = RegexpTokenizer(r'\w+')
table = str.maketrans('','','!\"#$%&\'()*+,./:;<=>?@[\]^_`{|}~')
corpus_combined = [] # used for vocabulary and average document length calculation
term_frequency = {}
data = []
term_frequency_unnormalized = {}
for doc in tqdm(corpus.keys()):
    temp = corpus[doc].lower().translate(table)
    temp = tokenizer.tokenize(temp)
    corpus[doc] = temp
    corpus_combined+= corpus[doc]
    data.append(temp)
    term_frequency_unnormalized[doc] = Counter(corpus[doc]) # This will be used later for BM25 calculation. It only contains the count.

#   Dictionary term_frequency contains key as doc_id and value is another dictionary where the terms of the 
#   documents are keys and term frequency are the values.
    term_frequency[doc] = {key:term_frequency_unnormalized[doc][key]/term_frequency_unnormalized[doc].most_common(1)[0][1] for key in term_frequency_unnormalized[doc].keys()}

vocab = set(corpus_combined)
Total_docs = len(corpus.keys())
idf = {}
idf_bm25 = {} #BM25 uses slightly different variant of IDF
negative_idfs = {} # Again used by BM25
# Constants for BM25 calculation. These values are same as used by Gensim Library
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25
# collect idf sum to calculate  average idf which will be used later to solve negative IDF values in BM25. Applies only to BM25
idf_sum = 0

for term in tqdm(vocab):
  count_doc = 0
  for doc in corpus.keys():
    if term in term_frequency[doc]:
      count_doc+=1
  idf[term] = math.log(Total_docs/count_doc)
  idf_bm25[term] = math.log(Total_docs - count_doc + 0.5) - math.log(count_doc + 0.5) # Definition of IDF used by BM25
  idf_sum += idf_bm25[term]
  if idf_bm25[term] < 0:
    negative_idfs[term] = 'random value' # just to keep track of terms having negative idfs
average_idf = float(idf_sum) / len(list(idf_bm25.keys()))
eps = EPSILON * average_idf
for term in tqdm(negative_idfs):
  idf_bm25[term] = eps

# Extracting Queries. Dictionary queries will contain query number as key and value will be another dictionary which will contain
# query term as keys and term frequency as values
queries = {}
with open('test_questions.txt', 'r') as f:  
    file = f.read()
root = BeautifulSoup(file, 'lxml')
for num, query in zip(list(range(1,101)),root.body.find_all('top')):
  text =  ' '.join(query.num.desc.text.strip().split()[1:])
  text = text.lower().translate(table)
  text = tokenizer.tokenize(text)
  temp = Counter(text)
  queries[num] = {key:temp[key]/temp.most_common(1)[0][1] for key in temp.keys()}

# Extracting pattern
with open('patterns.txt', 'r') as f:  
        file = f.read()
#file = file.lower()
file = file.split('\n')

query_answer = {} # This will contain query number as key and list of patterns as values. 

for line in file:
    if query_answer.get(int(line.split()[0])) == None:
      
       query_answer[int(line.split()[0])] = [' '.join(line.split()[1:])]
    else:
      
       query_answer[int(line.split()[0])].append(' '.join(line.split()[1:]))

score = retrieval_tfidf()
score_50 = {} # This will contain query number as key as and list of top 50 doc_ids 
#               for the corresponding query as values sorted in decreasing order of cosine_scores

score_1000 = {} # same as above but prints 1000 doc_ids. Later used by BM25 and ranksvm
for key in score.keys():
  score[key]  =sorted(score[key].items(), key=lambda x: x[1], reverse=True)
  score_50[key] = score[key][:50]
  score_1000[key] = score[key][:1000]

precision_50 = precision(score_50) # This contains the query number as key and precision@50 for the corresponding query as value
print('\nAvg Precision@50 of the baseline model is {}'.format(sum(list(precision_50.values()))/len(list(precision_50.values()))))
print('Precision@50 query wise of baseline model is\n{}'.format(precision_50))

# Uncomment the following block to print top most 50 relevant documents for queries along with scores:
'''
print('For the baseline model the scores along with doc_ids for each query are:\n')
for query in score_50.keys():
  print('Query number: {} score: {}'.format(query,score_50[query]))

'''

avg_dl =    len(corpus_combined)/Total_docs
score_bm25 = retrieval_BM25()
score_bm25_50 = {}
for key in score_bm25.keys():
    score_bm25_50[key]  =sorted(score_bm25[key].items(), key=lambda x: x[1], reverse=True)[:50]

precision_50_bm25 = precision(score_bm25_50)
print('\nAvg Precision@50 of BM25 is {}'.format(sum(list(precision_50_bm25.values()))/len(list(precision_50_bm25.values()))))
print('\nPrecision@50 query wise of BM25 is\n{}'.format(precision_50_bm25))

# Uncomment the following block to print top most 50 relevant documents for all queries along with scores:
'''
print('For the BM25 model the scores along with doc_ids for each query are:\n')
for query in score_bm25_50.keys():
  print('Query number: {} score: {}'.format(query,score_bm25_50[query]))

'''

#Training the SVM. We have used fold 1 of MQ2008 dataset of LETOR 4.0 benchmark datasets. The data is trained on five features:
# Document Length (Normalised by average document length), Term Frequency, TFIDF, BM25, IDF of query terms normalised by total terms in query
data = np.genfromtxt('dataset.csv',delimiter=',')
y = data[:,[-1,-2]] # The last two columns contain the relevance ordering of the document and queryid. 
#                     We compare documents pairwise for training only if they have same query id
y = y.astype('int32')
X= data[:,0:-2]
X_trans,y_trans = transform(X,y)
clf = LinearSVC()
clf.fit(X_trans,y_trans)

score_svm_50 = retrieval_SVM()
precision_50_svm = precision(score_svm_50)
print('\nAvg Precision@50 of SVM is {}'.format(sum(list(precision_50_svm.values()))/len(list(precision_50_svm.values()))))
print('\nPrecision@50 query wise of SVM is\n{}'.format(precision_50_svm))

#Uncomment the following block to print top most 50 relevant documents for queries along with scores:
'''
print('For the SVM model the scores along with doc_ids for each query are:\n')
for query in score_svm_50.keys():
  print('Query number: {} score: {}'.format(query,score_svm_50[query]))
'''



# Sentence ranker for BM25
sentence_raw,sentence_tokenized,avg_sl = get_sentence(score_bm25_50)
score_sentence_bm25 = retrieval_BM25_sentence(sentence_raw,sentence_tokenized,avg_sl)

score_sentence_bm25_50 = {} # for extracting top 50 sentences
for key in score_sentence_bm25.keys():
  score_sentence_bm25_50[key]=sorted(score_sentence_bm25[key].items(), key=lambda x: x[1], reverse=True)[:50]
mrr, ranklist_bm25 = MRR(score_sentence_bm25_50)
print('\nMean Reciprocal Rank of BM25 is {}'.format(mrr))
print('\nRank of first relevant sentence for each query using BM25 is:\n{}'.format(ranklist_bm25))

# Uncomment the following block if you want to print top 50 ranked sentences for each query along with score

'''
sentence_bm25_50  = {} # This will contain content of sentence as key instead of sentence_id
for query in score_sentence_bm25_50.keys():
  sentence_bm25_50[query] = [(sentence_raw[query][id],score) for id,score in score_sentence_bm25_50[query] ]
  print('Top 50 sentences for query {} are {}:\n '.format(query,sentence_bm25_50[query])) 
'''

# Sentence ranker for SVM
sentence_raw,sentence_tokenized,avg_sl = get_sentence(score_svm_50)
score_sentence_bm25 = retrieval_BM25_sentence(sentence_raw,sentence_tokenized,avg_sl)
score_sentence_svm_50 = retrieval_SVM_sentence(sentence_raw,sentence_tokenized,avg_sl,score_sentence_bm25)

mrr, ranklist_svm = MRR(score_sentence_svm_50)
print('\nMean Reciprocal Rank of SVM is {}'.format(mrr))
print('\nRank of first relevant sentence for each query using SVM is:\n{}'.format(ranklist_svm))

# Uncomment the following block if you want to print top 50 ranked sentences for each query along with score
'''
sentence_svm_50  = {}
for query in score_sentence_svm_50.keys():
  sentence_svm_50[query] = [(sentence_raw[query][id],score) for id,score in score_sentence_svm_50[query] ]
  print('Top 50 sentences for query {} is {}:\n '.format(query,sentence_svm_50[query])) 

'''

# Kendall rank correlation coefficient
svm_base = []
bm25_base = []
bm25_svm = []

for query in queries.keys():
  bm25 = [x[0] for x in score_bm25_50[query]]
  svm = [x[0] for x in score_svm_50[query]]
  basemodel = [x[0] for x in score_50[query]]
  tau,p_value = stats.kendalltau(bm25, svm)
  bm25_svm.append((tau,p_value))
  tau,p_value = stats.kendalltau(bm25, basemodel)
  bm25_base.append((tau,p_value))
  tau,p_value = stats.kendalltau(svm, basemodel)
  svm_base.append((tau,p_value))

print('Average kendall rank correlation coeffcient between basemodel and bm25 is {}'.format(sum([x[0] for x in bm25_base])/len(queries.keys())))
print('Average pvalue  between basemodel and bm25 is {}'.format(sum([x[1] for x in bm25_base])/len(queries.keys())))
#********************************************************************************************************************************************
print('Average kendall rank correlation coeffcient between basemodel and svm is {}'.format(sum([x[0] for x in svm_base])/len(queries.keys())))
print('Average pvalue between basemodel and svm is {}'.format(sum([x[1] for x in svm_base])/len(queries.keys())))
##############################################################################################################################################
print('Average kendall rank correlation coeffcient between svm and bm25 is {}'.format(sum([x[0] for x in bm25_svm])/len(queries.keys())))
print('Average pvalue between svm and bm25 is {}'.format(sum([x[1] for x in bm25_svm])/len(queries.keys())))

# Printing total queries for which atleast one relevant doc was found for all the models:
print('Total queries having atleast one relevant doc in top 50 for baseline model {}'.format(len([x for x in precision_50.values() if x!=0])))
print('Total queries having atleast one relevant doc in top 50 for bm25 model {}'.format(len([x for x in precision_50_bm25.values() if x!=0])))
print('Total queries having atleast one relevant doc in top 50 for SVM {}'.format(len([x for x in precision_50_svm.values() if x!=0])))

print('Total queries having relevant sentence at rank 1 for BM25 is {}'.format(len([x for x in ranklist_bm25.values() if x==1])))
print('Total queries having relevant sentence at rank 1 for RankSVM is {}'.format(len([x for x in ranklist_svm.values() if x==1])))
print('Total queries having no relevant sentence in top 50 for BM25 is {}'.format(len([x for x in ranklist_bm25.values() if x==None])))
print('Total queries having no relevant sentence in top 50 for SVM is {}'.format(len([x for x in ranklist_svm.values() if x==None])))