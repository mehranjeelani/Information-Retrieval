# snlp-final-project
Final Project for Statistical Natural Language Processing Course
### Tasks
1. Baseline Document Retreival Model
   - [x] Extract text from corpus
   - [x] Preprocess the texts from corpus and apply tokenisation
   - [x] Compute idf
   - [x] Comput tf
   - [x] Give list of query terms as product of term's idf and tf-value
   - [x] Relavance based on cosine similarity
   - [x] Sort similarity scores and output top 50 most relevant documents
   - [x] Function to evaluate performance of document using precision at r with r = 50
   - [x] Test on test_questions.txt
2. Advanced Document Retriever with Re-Ranking
   - [x] Use the baseline model and return the top 1000 documents
   - [x] Re-rank the top 1000 documents with a more advanced approach
3. Sentence Ranker
   - [x] Split the top 50 documents into sentences (sent_tokenize)
   - [x] Treat the sentences likedocuments to rank them and return the top 50 sentences (same approach as above)
   - [x] Evaluate performance using Mean Reciprocal Rank
