# snlp-final-project
Final Project for Statistical Natural Language Processing Course
### Tasks
1. Baseline Document Retreival Model
   - [x] Extract text from corpus
   - [x] Preprocess the texts from corpus and apply tokenisation
   - [x] Compute idf
   - [x] Comput tf
   - [ ] Give list of query terms as product of term's idf and tf-value
   - [ ] Relavance based on cosine similarity
   - [ ] Sort similarity scores and output top 50 most relevant documents
   - [ ] Function to evaluate performance of document using precision at r with r = 50
   - [ ] Test on test_questions.txt
2. Advanced Document Retriever with Re-Ranking
   - [ ] Use the baseline model and return the top 1000 documents
   - [ ] Re-rank the top 1000 documents with a more advanced approach
3. Sentence Ranker
   - [ ] Split the top 50 documents into sentences (sent_tokenize)
   - [ ] Treat the sentences likedocuments to rank them and return the top 50 sentences (same approach as above)
   - [ ] Evaluate performance using Mean Reciprocal Rank
