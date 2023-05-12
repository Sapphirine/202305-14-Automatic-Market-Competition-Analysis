import os
import json
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import seaborn as sns
import tensorflow as tf
import seaborn as sns
import matplotlib
matplotlib.use('agg')
# import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import shutil
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import umap.umap_ as umap # dimensionality reduction
import hdbscan # clustering
from functools import partial
from collections import Counter

# To perform the Bayesian Optimization for searching the optimum hyperparameters, 
# we use hyperopt package:
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
import collections
import spacy
from spacy import displacy
from collections import Counter
from wordcloud import WordCloud 

sns.set_style("darkgrid")


nltk.download('stopwords')
nltk.download('punkt')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def embed(model, model_type, sentences):
    if model_type == 'use':
        embeddings = model(sentences)
    elif model_type == 'sentence transformer':
        embeddings = model.encode(sentences)
    
    return embeddings

def download_if_missing(url, file_path, extract=True):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, file_path)
    if os.path.exists(file_path):
        return file_path
    return tf.keras.utils.get_file(file_path, origin=url, extract=extract)

def get_labels(data):
  unique_labels = []
  all_labels = []

  for labels in data:
    cur_label = []
    for label in labels:
      cur_label.append(label)
      if label not in unique_labels:
        unique_labels.append(label)
    all_labels.append(cur_label)
    return unique_labels, all_labels

def replace_hearts_with_PAD(text):
    return re.sub(r"[♥]+", ' **** ' ,text)

def small_clean(text):
    text = re.sub('<[^<]+?>', '', text)
    text = text.replace('&quot', '').replace('\r\n', '')
    text = re.sub('\s+', ' ', text) # replace multiple whitespaces with a single whitespace

    return text
def initial_clean(text):
    """
    Function to clean text of html, websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub('<[^<]+?>', '', text)
    text = text.replace('&quot', '').replace('\r\n', '')
    #text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text
stop_words = stopwords.words('english')
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]

def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    try:
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] 
    except IndexError: 
        pass
    return text

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))


def train_lda(data):
    """
    This function trains the lda model
    We setup parameters like number of topics, the chunksize to use in Hoffman method
    We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
    """
    num_topics = 5
    chunksize = 150
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]

    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=1e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)

    return dictionary,corpus,lda

def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    p = query[None,:].T 
    q = matrix.T 
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))


def get_most_similar_documents(query,matrix,k=10):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances

def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      min_samples = None,
                      random_state = None):
    """
    Returns HDBSCAN objects after first performing dimensionality reduction using UMAP
    
    Arguments:
        message_embeddings: embeddings to use
        n_neighbors: int, UMAP hyperparameter n_neighbors
        n_components: int, UMAP hyperparameter n_components
        min_cluster_size: int, HDBSCAN hyperparameter min_cluster_size
        min_samples: int, HDBSCAN hyperparameter min_samples
        random_state: int, random seed
        
    Returns:
        clusters: HDBSCAN object of clusters
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors = n_neighbors, 
                                n_components = n_components, 
                                metric = 'cosine', 
                                random_state=random_state)
                            .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, 
                               min_samples = min_samples,
                               metric='euclidean', 
                               gen_min_span_tree=True,
                               cluster_selection_method='eom').fit(umap_embeddings)
    
    return clusters

def score_clusters(clusters, prob_threshold = 0.05):
    """
    Returns the label count and cost of a given clustering

    Arguments:
        clusters: HDBSCAN clustering object
        prob_threshold: float, probability threshold to use for deciding
                        what cluster labels are considered low confidence

    Returns:
        label_count: int, number of unique cluster labels, including noise
        cost: float, fraction of data points whose cluster assignment has
              a probability below cutoff threshold
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    
    return label_count, cost

def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize

    Arguments:
        params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'random_state' and
               their values to use for evaluation
        embeddings: embeddings to use
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters

    Returns:
        loss: cost function result incorporating penalties for falling
              outside desired range for number of clusters
        label_count: int, number of unique cluster labels, including noise
        status: string, hypoeropt status

        """
    
    clusters = generate_clusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 random_state = params['random_state'])
    
    label_count, cost = score_clusters(clusters, prob_threshold = 0.05) # 0.05
    
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.5 #0.5 
    else:
        penalty = 0
    
    loss = cost + penalty
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}

def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

        """
    
    trials = Trials()
    fmin_objective = partial(objective, 
                             embeddings=embeddings, 
                             label_lower=label_lower,
                             label_upper=label_upper)
    
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=max_evals, 
                trials=trials)

    best_params = space_eval(space, best)
    print ('best:')
    print (best_params)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    
    best_clusters = generate_clusters(embeddings, 
                                      n_neighbors = best_params['n_neighbors'], 
                                      n_components = best_params['n_components'], 
                                      min_cluster_size = best_params['min_cluster_size'],
                                      random_state = best_params['random_state'])
    
    return best_params, best_clusters, trials


# Define hyperparameter search space
hspace = {
    "n_neighbors": hp.choice('n_neighbors', range(3,32)),
    "n_components": hp.choice('n_components', range(3,32)),
    "min_cluster_size": hp.choice('min_cluster_size', range(2,32)),
    "random_state": 42
}

label_lower = 10
label_upper = 100
max_evals = 5 


nlp = spacy.load("en_core_web_sm")


def compute_IDF(documents):
    word_count = Counter()
    for doc in documents:
        # if 'drops(players' in doc:
        #     print(doc)
        #     print(doc.split())
        words_set = set(doc.split())
        word_count.update(words_set)
    total = sum(word_count.values())
    return {k: round((np.log2(total / v)))  for k, v in word_count.items()} 

def get_group(df, category_col, category):
    """
    Returns documents of a single category
    
    Arguments:
        df: pandas dataframe of documents
        category_col: str, column name corresponding to categories or clusters
        category: int, cluster number to return
    Returns:
        single_category: pandas dataframe with documents from a single category
    """
    
    single_category = df[df[category_col]==category].reset_index(drop=True)

    return single_category 

def most_common(lst, n_words, word_IDF):
    """
    Get most common words in a list of words
    
    Arguments:
        lst: list, each element is a word
        n_words: number of top common words to return
    
    Returns:
        counter.most_common(n_words): counter object of n most common words
    """
    counter=collections.Counter(lst)
    
    for k in list(counter): 
        if counter[k] ==1: 
            pass 
        else:
            counter[k] *= word_IDF[k] 
        
    return counter.most_common(n_words)

def extract_labels(category_docs, word_IDF, print_word_counts=False):
    """
    Extract labels from documents in the same cluster by concatenating
    most common verbs, ojects, and nouns

    Argument:
        category_docs: list of documents, all from the same category or
                       clustering
        print_word_counts: bool, True will print word counts of each type in this category

    Returns:
        label: str, group label derived from concatentating most common
               verb, object, and two most common nouns

    """

    verbs = []
    dobjs = []
    nouns = []
    adjs = []
    
    verb = ''
    dobj = ''
    noun1 = ''
    noun2 = ''

    
    for i in range(len(category_docs)):
        doc = nlp(category_docs[i])
        for token in doc:
            if (token.is_stop==False) and (len(str(token).strip()) > 0): 
                if token.pos_ == 'VERB':
                    verbs.extend([token.lemma_.lower()]) 

                elif token.dep_=='dobj':
                    dobjs.extend([token.lemma_.lower()]) 

                elif token.pos_=='NOUN':
                    nouns.extend([token.lemma_.lower()]) 
                    
                elif token.pos_=='ADJ':
                    adjs.extend([token.lemma_.lower()])

    if print_word_counts:
        for word_lst in [verbs, dobjs, nouns, adjs]:
            counter=collections.Counter(word_lst)
            print(counter)
    
    if len(verbs) > 0:
        verb = most_common(verbs, 1, word_IDF)[0][0]
    
    if len(dobjs) > 0:
        dobj = most_common(dobjs, 1, word_IDF)[0][0]
    
    if len(nouns) > 0:
        noun1 = most_common(nouns, 1, word_IDF)[0][0]
    
    if len(set(nouns)) > 1:
        noun2 = most_common(nouns, 2, word_IDF)[1][0]
    
    label_words = [verb, dobj]
    
    for word in [noun1, noun2]:
        if word not in label_words:
            label_words.append(word)
    
    if '' in label_words:
        label_words.remove('')
    
    label = '_'.join(label_words)
    
    return label

def apply_and_summarize_labels(df, word_IDF, category_col):
    """
    Assign groups to original documents and provide group counts

    Arguments:
        df: pandas dataframe of original documents of interest to
            cluster
        category_col: str, column name corresponding to categories or clusters

    Returns:
        summary_df: pandas dataframe with model cluster assignment, number
                    of documents in each cluster and derived labels
    """
    
    numerical_labels = df[category_col].unique()
    
    label_dict = {}
    for label in numerical_labels:
        current_category = list(get_group(df, category_col, label)['text'])
        label_dict[label] = extract_labels(current_category, word_IDF)
        
    summary_df = (df.groupby(category_col)['text'].count()
                    .reset_index()
                    .rename(columns={'text':'count'})
                    .sort_values('count', ascending=False))
    
    summary_df['label'] = summary_df.apply(lambda x: label_dict[x[category_col]], axis = 1)
    
    return summary_df

def WordCloud_generator(data, path, title=None):
    
    # Keep top 1000 most frequent words
    most_freq = Counter(data).most_common(1000) 
    text = ' '.join([x[0] for x in most_freq])
    
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='white',
                          min_font_size = 10,
                          collocations=False
                         ).generate(text)

    # plot the Word Cloud    
    plt.clf()                  
    plt.figure(figsize = (6, 6), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=25)
    plt.savefig(path)
    plt.clf()

class SteamData:
    def __init__(self, root='data'):
        # prepare data
        self.root = root
        self.download_data()
        self.clean_data()
        print("="*50)
        print("Data ready!")
        print("="*50)

        
    def download_data(self):
        steam_path = os.path.join(self.root, "steam.csv")
        steam_path = download_if_missing("https://storage.googleapis.com/eecs6895-project/steam.csv",
                            steam_path)

        steam_description_path = os.path.join(self.root, "steam_description_data.csv")
        steam_description_path = download_if_missing("https://storage.googleapis.com/eecs6895-project/steam_description_data.csv",
                            steam_description_path)
        
        review_path = os.path.join(self.root, "reviews.csv")
        review_path = download_if_missing("https://storage.googleapis.com/eecs6895-project2/reviews.csv",
                            review_path)

        self.steam = pd.read_csv(steam_path)
        self.steam_description = pd.read_csv(steam_description_path)
        self.df_reviews = pd.read_csv(review_path)

        
    def clean_data(self):
        self.steam.loc[self.steam['english'] != 1].index
        self.steam = self.steam.drop(self.steam.loc[self.steam['english'] != 1].index)


        # Clean categories to lower cases for further analysis
        self.steam['categories'] = self.steam['categories'].apply(lambda x: x.split(";"))
        self.steam['categories'] = self.steam['categories'].apply(lambda x: list(map(str.lower, x)))
        
        # Clean the tag values into lists and lower cases
        self.steam['steamspy_tags'] = self.steam['steamspy_tags'].apply(lambda x: x.split(";"))
        self.steam['steamspy_tags'] = self.steam['steamspy_tags'].apply(lambda x: list(map(str.lower, x)))

        # Clean the Genre values into lists and lower case
        self.steam['genres'] = self.steam['genres'].apply(lambda x: x.split(";"))
        self.steam['genres'] = self.steam['genres'].apply(lambda x: list(map(str.lower, x)))
        
        # convert review text to string
        self.df_reviews["review_text"] = self.df_reviews["review_text"].astype(str)
        self.df_reviews["review_votes"] = self.df_reviews["review_votes"].astype(str)
        self.df_reviews.review_text = self.df_reviews.review_text.apply(lambda s: s.strip())

        # drop the reviews with null score
        self.df_reviews = self.df_reviews[self.df_reviews["review_score"].notnull()]

        # change the scores from 1, -1 to 1 and 0
        self.df_reviews["review_score"] = \
        np.where(self.df_reviews["review_score"]==-1, 0, self.df_reviews["review_score"])


        # Remove early access comments
        # These are the reviews with no comments writen by a human/reviewer. 
        self.df_reviews = self.df_reviews[self.df_reviews.review_text != "Early Access Review"]
        self.df_reviews = self.df_reviews[~self.df_reviews.review_text.isin(['nan'])]

        # Drop duplicates if there is any
        self.df_reviews.drop_duplicates(['review_text', 'review_score'], inplace = True)
        
        # Text cleaning: replace heart, which originally represents F words, with '**' to improve embedding accuracy since it will impact the classifier in negative reviews
        self.df_reviews['review_text_clean'] = self.df_reviews.review_text.apply(replace_hearts_with_PAD)

        
    def user_input(self, category, tag, genre):
        if os.path.exists('result'):
            shutil.rmtree('result')
        os.makedirs('result')
        self.user_category = category
        self.user_tag = tag
        self.user_genre = genre
        
        user_index = self.steam.loc[(self.steam['categories'].map(lambda x: self.user_category in x)) &
        (self.steam['steamspy_tags'].map(lambda x: self.user_tag in x)) &
        (self.steam['genres'].map(lambda x: self.user_genre in x))].index
        self.user_steam = self.steam.loc[user_index]
        
        if user_index.empty:
            print("No game found, please try again.")
            return False
        
        self.user_steam_desc = pd.merge(self.user_steam, self.steam_description, left_on = 'appid', right_on = 'steam_appid', how = 'left')
        
        self.user_steam_desc.loc[self.user_steam_desc['detailed_description'].isna()].index
        self.user_steam_desc.drop(self.user_steam_desc.loc[self.user_steam_desc['detailed_description'].isna()].index)
        self.user_steam_desc['tokenized'] = self.user_steam_desc['detailed_description'].apply(apply_all)
        return True
        

    def get_word_frequency(self):
        all_words = [word for item in list(self.user_steam_desc['tokenized']) for word in item]
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)

        k = 5000
        # define a function only to keep words in the top k words
        top_k_words,_ = zip(*fdist.most_common(k))
        top_k_words = set(top_k_words)
        frequent_words = set(['game', 'play', 'player', 'new'])
        
        def keep_top_k_words(text):
            return [word for word in text if word in top_k_words and word not in frequent_words]


        self.user_steam_desc['tokenized'] = self.user_steam_desc['tokenized'].apply(keep_top_k_words)


        # document length
        self.user_steam_desc['doc_len'] = self.user_steam_desc['tokenized'].apply(lambda x: len(x))
        self.user_steam_desc.drop(labels='doc_len', axis=1, inplace=True)

    def drop_short_articles(self):
        self.user_steam_desc = self.user_steam_desc[self.user_steam_desc['tokenized'].map(len) >= 30]
        self.user_steam_desc = self.user_steam_desc[self.user_steam_desc['tokenized'].map(type) == list]
        self.user_steam_desc.reset_index(drop=True,inplace=True)
        
    def train_test_split(self):
        msk = np.random.rand(len(self.user_steam_desc)) < 0.99

        self.train_df = self.user_steam_desc[msk]
        self.train_df.reset_index(drop=True,inplace=True)

        self.test_df = self.user_steam_desc[~msk]
        self.test_df.reset_index(drop=True,inplace=True)
    
    def train_lda(self):
        self.dictionary,self.corpus,self.lda = train_lda(self.train_df)
        
        random_article_index = np.random.randint(len(self.train_df))
        bow = self.dictionary.doc2bow(self.train_df.iloc[random_article_index]['tokenized'])
        self.doc_distribution = np.array([tup[1] for tup in self.lda.get_document_topics(bow=bow)])
        
        random_article_index = np.random.randint(len(self.test_df))
        new_bow = self.dictionary.doc2bow(self.test_df.iloc[random_article_index]['tokenized'])
        self.new_doc_distribution = np.array([tup[1] for tup in self.lda.get_document_topics(bow=new_bow)])
        
        self.doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in self.lda[self.corpus]])
        
    def get_most_similar_documents(self):
        most_sim_ids = get_most_similar_documents(self.new_doc_distribution,self.doc_topic_dist)

        self.most_similar_df = self.train_df[self.train_df.index.isin(most_sim_ids)]
        # self.most_similar_df = self.train_df.iloc[most_sim_ids]

        self.most_similar_game_names = self.most_similar_df['name'][:5]
        self.most_similar_game_descriptions = (self.most_similar_df
        ['short_description'][:5]).apply(small_clean)
        self.app_ids = list(self.most_similar_df['appid'])

        for name, description in zip(self.most_similar_game_names, self.most_similar_game_descriptions):
            print(name)
            print(description)
            print('------------------------')
            
    def plt1(self):
        plt.figure(figsize=(10, 6)) 
        plt.clf()
        ax = sns.histplot(data=self.most_similar_df['price'])
        ax.axvline(x=np.mean(self.most_similar_df['price']), color='r')
        plt.text(np.mean(self.most_similar_df['price'])+0.1,5.5,'mean',rotation=0)
        ax.axvline(x=np.median(self.most_similar_df['price']), color='b')
        plt.text(np.median(self.most_similar_df['price'])+0.1,5.5,'median',rotation=0)
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title("Price Distribution of Direct Competitors")
        plt.autoscale() 
        plt.subplots_adjust(bottom=0.15) 
        plt.savefig('result/game_statistics_analysis_1.png')
        plt.clf()
        
    def plt2(self):
        plt.figure(figsize=(10, 6)) 
        plt.clf()
        ax = sns.histplot(data=self.user_steam_desc['price'])
        ax.axvline(x=np.mean(self.user_steam_desc['price']), color='r')
        plt.text(np.mean(self.user_steam_desc['price'])+1,350,'mean',rotation=0)
        ax.axvline(x=np.median(self.user_steam_desc['price']), color='b')
        plt.text(np.median(self.user_steam_desc['price'])-6.5,350,'median',rotation=0)
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title("Price Distribution of Indirect Competitors")
        plt.autoscale() 
        plt.savefig('result/game_statistics_analysis_2.png')
        plt.clf()
        
    def plt3(self):
        plt.figure(figsize=(10, 6)) 
        plt.clf()
        sns.countplot(y=self.user_steam_desc['owners'])
        plt.xlabel('Owner Count')
        plt.ylabel('Game Count')
        plt.title("Owner Distribution of Indirect Competitors")
        plt.autoscale() 
        plt.savefig('result/game_statistics_analysis_3.png')
        plt.clf()
    
    def plt4(self):
        plt.figure(figsize=(10, 6)) 
        plt.clf()
        sns.countplot(y=self.most_similar_df['owners']) 
        plt.xlabel('Owner Count')
        plt.ylabel('Game Count')
        plt.title("Owner Distribution of Direct Competitors")
        plt.autoscale() 
        plt.savefig('result/game_statistics_analysis_4.png')
        plt.clf()
    def game_statistics_analysis(self):
        self.plt1()
        self.plt2()
        self.plt3()
        self.plt4()
        

        
    def user_review_analysis(self):
        self.select_reviews(0)
        self.select_reviews(1)
        
    def select_reviews(self, score):
        # Select negative reviews
        reviews = self.df_reviews[(self.df_reviews.app_id.isin(self.app_ids)) & (self.df_reviews.review_score == score)]
        #neg_reviews = neg_reviews.sample(n=2000, random_state = 1234)
        all_intents = reviews.review_text_clean.tolist()

        # Split reviews into sentences
        # Remove sentences with less than 4 words
        all_sents = []
        for intent in all_intents:
            for sent in nltk.sent_tokenize(intent):
                if len(sent.split()) > 4:
                    all_sents.append(sent)
        all_intents = all_sents
        
        model_st1 = SentenceTransformer('all-mpnet-base-v2')

        embeddings_st1 = embed(model_st1, 'sentence transformer', all_intents)
        
        best_params_use, best_clusters_use, trials_use = bayesian_search(embeddings_st1, space=hspace, label_lower=label_lower, label_upper=label_upper, max_evals=max_evals)
        
        data_clustered = pd.DataFrame(data = list(zip(all_intents,best_clusters_use.labels_)),
                             columns = ['text', 'label_st1'])
        
        sent_with_word_lemma = []
        for intent in all_intents:
            doc = nlp(intent)
            sent_temp = ""
            this_one = False
            for token in doc:
                if (token.pos_ in ['VERB', 'NOUN', 'ADJ']) or (token.dep_=='dobj'):
                    sent_temp += token.lemma_.lower() + " "
            sent_with_word_lemma.append(sent_temp)
        word_IDF = compute_IDF(sent_with_word_lemma)
        cluster_summary = apply_and_summarize_labels(data_clustered, word_IDF, 'label_st1')
        pd.set_option('display.max_rows', None)

        C_ = cluster_summary.iloc[0,0]
        selected_reviews_1 = []
        for index, clust in enumerate(best_clusters_use.labels_):
            if clust == C_:
                selected_reviews_1.append(all_intents[index])
            
        C_ = cluster_summary.iloc[1,0]
        selected_reviews_2 = []
        for index, clust in enumerate(best_clusters_use.labels_):
            if clust == C_:
                selected_reviews_2.append(all_intents[index])
            
        stopwords = ['game', 'play', 'games', 'played']
        for i in range(len(selected_reviews_1)):
            sentence = selected_reviews_1[i].split()
            sentence = [word for word in sentence if (word not in stopwords and word not in stop_words)]
            selected_reviews_1[i] = ' '.join(sentence)

        WordCloud_generator(selected_reviews_1, f'result/score{score}_cloud1.png',"Most Used Words in Topic 'play_achievement_level_boss'")
        
        for i in range(len(selected_reviews_2)):
            sentence = selected_reviews_2[i].split()
            sentence = [word for word in sentence if (word not in stopwords and word not in stop_words)]
            selected_reviews_2[i] = ' '.join(sentence)

        WordCloud_generator(selected_reviews_2, f'result/score{score}_cloud2.png', "Most Used Words in Topic 'want skill tree'")
        
        return (cluster_summary['label'][:10]).tolist()


            
if __name__ == '__main__':
    user_category = 'single-player'
    user_tag = 'indie'
    user_genre = 'strategy'
    steam = SteamData()
    steam.user_input(user_category, user_tag, user_genre)