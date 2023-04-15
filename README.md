# Conversational Collaborative Filtering using External Data and MovieSent dataset

Code and data from the NAACL'21 short paper ["You Sound Like Someone Who Watches Drama Movies: Towards Predicting Movie Preferences from Conversational Interactions" by Volokhin et al.](https://www.aclweb.org/anthology/2021.naacl-main.246/)

*MovieSent* - dataset containing 489 movie-related conversations with fine-grained user sentiment labels about each mentioned movie.
Conversations are in the [MovieSent.json](data/MovieSent.json) file.

Added https://research.google/resources/datasets/coached-conversational-preference-elicitation/ and combined the data from Moviesent and created the file merged_data.json in the data folder.

Reviews were collected in April 2020. Initially a list of critics is compiled from more than 600 movies, their IDs are in [films_rt_ids.json](data/films_rt_ids.json). Then for those critics all their reviews are scraped and put into [reviews.tar.gz](data/reviews.tsv.gz) file. 

To run the model:

1) Install [requirements.txt](requirements.txt)
2) Run [indexing.py](indexing.py) to create an index of reviews based on the [reviews.tsv.gz](data/reviews.tsv.gz) file.
3) Run [sentiment_estimation.py](sentiment_estimation.py) to create a sentiment estimation model.
4) Run [main.py](main.py) for the final model. Training of CF model will occur at the same time, and can take a long time for a SVDpp model (KNN is much faster, ~20 seconds, if you just want to check if the code works).

Added a new model and you can fill in NMF to run the Neural Matrix Factorization in the dataset.py file on line 294.
