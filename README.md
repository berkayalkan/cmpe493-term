# Information Retrieval for Covid-19
In this project, we worked on information retrieval from the Covid-19 related scientific
literature. The task can be found in [TREC-COVID Challange](https://ir.nist.gov/covidSubmit/).
## Notes
- This project developed in a MacOs & Windows machine using Python 3.8.5.
- Required packages can be seen in the "requirements.txt".
- "metadata.csv" and "official_results.txt" should be located in the ".input" directory. Files can be found [here](https://drive.google.com/drive/folders/1ghTgJ9_tge6WgR1wfnAXDU2LbDU5MZZC?usp=sharing)
- Any kinds of outputs will be saved in the ".output" folder, after execution.
## Dependencies
 | package | version |
 | ------- | ------- |
 | nltk    |   3.5   |
 | gensim  |   3.8.3 |
 | tqdm    |   4.56.0|
 | gensim  |   3.8.3 |
 | beautifulsoup4|	4.9.3 |
 | requests	| 2.25.1 |
 | sent2vec | 0.2.0 |
 | rank-bm25 | 0.2.1 |
##  Execution
To run the program successfully, execute these commands mentioned below inside the project folder:
*  For TF-IDF and Doc2Vec:
```
python3 main.py
```
*  For BM25:
For creating BM25 model results in a pickle and to create results for TREC eval, run the following command:
```
python3 bm25/bm25_operations.py
```
*  For Rerank with BERT: <br/>
First run following command to create BERT similarities dictionary:
```
python3 bert/rerank_bert.py
```
To create relevant TREC eval files run these commands:
```
python3 bert/write_results_bert.py
python3 bert/bm25_top100_test_write_results.py
python3 bert/official_results_top100.py 
```
## Team
- [Sertay Akpinar](https://github.com/sertayy)
- [Olcayto Türker](https://github.com/olcaytoturker)
- [Berkay Alkan](https://github.com/berkayalkan)

## License
[BSD 3-Clause License](https://github.com/berkayalkan/cmpe493-term/blob/master/LICENSE)
