# Reference: https://github.com/bpw1621/streamlit-topic-modeling/blob/master/streamlit_topic_modeling/app.py

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import pandas as pd

import re
import string
import contractions
import random
import nltk
import gensim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyLDAvis
import pyLDAvis.gensim

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from textblob import Word
from gensim import corpora
from gensim.models import CoherenceModel

import requests
from bs4 import BeautifulSoup

COLORS = [color for color in mcolors.XKCD_COLORS.values()]

# Functions for Web Scrapers ######################################################################
def linkedin_scraper(df, webpage, page_number):
    
    next_page = webpage + str(page_number)
    print (str(next_page))
    print(df.shape)
    response = requests.get(str(next_page))
    soup = BeautifulSoup(response.content,'html.parser')

    jobs = soup.find_all('div', class_='base-card relative w-full hover:no-underline focus:no-underline base-card--link base-search-card base-search-card--link job-search-card')
    for job in jobs:
        job_title = job.find('h3', class_='base-search-card__title').text.strip()
        job_company = job.find('h4', class_='base-search-card__subtitle').text.strip()
        job_location = job.find('span', class_='job-search-card__location').text.strip()
        job_link = job.find('a', class_='base-card__full-link')['href']
        
        response2 = requests.get(job_link)
        soup2 = BeautifulSoup(response2.content,'html.parser')
        job_desc = soup2.find('div', class_="show-more-less-html__markup show-more-less-html__markup--clamp-after-5")
        
        if (job_desc is not None):
            job_desc = job_desc.text.strip()
        
        df_new_row = pd.DataFrame({
            'title': [job_title],
            'company': [job_company],
            'description': [job_desc]
        })
        
        df = pd.concat([df, df_new_row])
        #print('Data updated')
        
    if page_number < 25:
        page_number = page_number + 25
        linkedin_scraper(df, webpage, page_number)
    else:
        df.to_csv('linkedin-jobs.csv', index=False)

def sgjobsdb_scraper(df, webpage, page_number):
    
    print(webpage + "&p=" + str(page_number))
    
    response = requests.get(webpage + "&p=" + str(page_number))
    soup = BeautifulSoup(response.content,'html.parser')

    jobs = soup.find_all('div', class_='job-card')
    for job in jobs:
        job_title = job.find('h3', class_='job-title').text.strip()
        job_company = job.find('span', class_='job-company').text.strip()
        job_link = job.find('a', class_='job-link')['href']
                
        response2 = requests.get("https://sg.jobsdb.com"+job_link)
        soup2 = BeautifulSoup(response2.content,'html.parser')
        job_desc = soup2.find('div', class_="z1s6m00 _5135ge0 _5135ge7 _5135gei")
        
        if (job_desc is not None):
            job_desc = job_desc.text.strip()
            
        else:
            job_desc = soup2.find('div', id="job-description-container")
            
            if (job_desc is not None):
                job_desc = job_desc.text.strip()
                    
        df_new_row = pd.DataFrame({
            'title': [job_title],
            'company': [job_company],
            'description': [job_desc]
        })
        
        df = pd.concat([df, df_new_row])

    if page_number < 20:
        page_number = page_number + 1
        sgjobsdb_scraper(df, webpage, page_number)
    else:
        df.to_csv('sgjobdb-jobs.csv', index=False)

# Functions for Text Preprocessing ######################################################################
def strip_links(text):
    link_regex    = re.compile('((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def remove_stop_words(sentence, common_words):
    stopwords_list=set(stopwords.words("english"))
    stopwords_list.update(common_words)
    
    # tokenisation
    word_list = sentence.split()
    cleaned_sentence = ' '.join([w for w in word_list if (w.lower() not in stopwords_list)])
    
    return (cleaned_sentence)

@st.cache_data()
def preprocess_text(texts_df: pd.DataFrame, text_column: str, common_words):
    texts_df[text_column] = texts_df[text_column].apply(lambda x: "".join(x.lower() for x in x))
    texts_df[text_column] = texts_df[text_column].apply(lambda x: strip_all_entities(strip_links(x)))
    texts_df[text_column] = texts_df[text_column].apply(lambda x: re.sub(r'\d+', '', x))
    texts_df[text_column] = texts_df[text_column].str.strip()
    texts_df[text_column] = texts_df[text_column].apply(lambda x: contractions.fix(x))
    texts_df[text_column] = texts_df[text_column].apply(remove_punctuations)
    texts_df[text_column] = texts_df[text_column].apply(lambda x: remove_stop_words(x,common_words))
    texts_df[text_column] = texts_df[text_column].str.strip()
    texts_df[text_column] = texts_df[text_column].apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))
    freq = pd.Series(' '.join(texts_df[text_column]).split()).value_counts()
    less_freq = list(freq[freq ==1].index)
    texts_df[text_column] = texts_df[text_column].apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
    texts_df[text_column] = texts_df[text_column].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    texts_df[text_column] = texts_df[text_column].apply(lambda x: word_tokenize(x))
    return texts_df[text_column].values.tolist()

@st.cache_data()
def create_bigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs, min_count=5)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    docs = [bigram_phraser[doc] for doc in docs]        
    return docs

def generate_N_grams(text,ngram=1):
    words=[word for word in text if word not in set(stopwords.words('english'))]  
    temp=zip(*[words[i:] for i in range(0,ngram)])
    ans=['_'.join(ngram) for ngram in temp]
    return ans

@st.cache_data()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus

@st.cache_data()
def train_model(docs, topic_num):
    id2word, corpus = prepare_training_data(docs)
    model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=topic_num, id2word=id2word,
                                            random_state=100, update_every=1, chunksize=100, 
                                            passes=10, alpha='auto', per_word_topics=True)
    return id2word, corpus, model

@st.cache_data()
def train_models(n_topics, corpus, _id2word):
    models=[]
    perplexity_scores=[]
    coherence_scores=[]

    LDA = gensim.models.ldamodel.LdaModel

    for i in n_topics:
        model = LDA(corpus=corpus, num_topics=i, id2word=id2word,
                    random_state=100, update_every=1, chunksize=100,
                    passes=10, alpha='auto', per_word_topics=True)
        models.append(model)
        perplexity_scores.append(calculate_perplexity(model, corpus))
        coherence_scores.append(calculate_coherence(model, corpus))
    
    return models, perplexity_scores, coherence_scores

def clear_session_state():
    for key in ('model_kwargs', 'id2word', 'corpus', 'model', 'previous_perplexity', 'previous_coherence_model_value'):
        if key in st.session_state:
            del st.session_state[key]

def calculate_perplexity(model, corpus):
    return np.exp2(-model.log_perplexity(corpus))

def calculate_coherence(model, corpus):
    coherence_model = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
    return coherence_model.get_coherence()

@st.cache_data()
def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud(width=700, height=600, background_color='white', 
                          collocations=collocations).generate(wordcloud_text)
    return wordcloud

##############################################################################################################
st.set_page_config(page_title='Job Skills Analysis', page_icon='./data/favicon.png', layout='wide')

st.title('Job Skills')

with st.form(key='keyword_form'):
    input_keyword = st.text_input('Enter keywords to scrape:')
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # scrap linkedin jobs
    with st.spinner('Scraping linkedin jobs... please wait...'):
        keyword = input_keyword.replace(' ','%20')
        linkedin_url = "https://www.linkedin.com/jobs/search?keywords=" + "&location=Singapore&geoId=102454443&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0&start="

        df = pd.DataFrame(columns=['title','company','description'])
        linkedin_scraper(df, linkedin_url, 0)

    # scrap sgjobsdb jobs
    with st.spinner('Scraping sgjobsdb jobs... please wait...'):
        keyword = input_keyword.replace(' ','+')
        sgjobsdb_url = "https://sg.jobsdb.com/j?sp=homepage&trigger_source=homepage&q=" + keyword

        df = pd.DataFrame(columns=['title','company','description'])
        sgjobsdb_scraper(df, sgjobsdb_url, 1)

    # combine dataset
    df_linkedin = pd.read_csv('linkedin-jobs.csv')
    df_linkedin.drop_duplicates(inplace=True)

    df_sgjobdb = pd.read_csv('sgjobdb-jobs.csv')
    df_sgjobdb.drop_duplicates(inplace=True)

    df_jobs = pd.concat([df_linkedin, df_sgjobdb])
    df_jobs.dropna(inplace=True) 
    
    df_jobs.to_csv('jobs.csv', index=False)

    st.success('Completed!', icon="✅")
    st.session_state.jobs = df_jobs

if 'jobs' not in st.session_state:
    st.stop()

#df_jobs = pd.read_csv('jobs.csv')
st.dataframe(df_jobs)

# Text Preprocessing Pipeline
common_words=['job','description', 'responsibilities', 'experience', 'years', 'year', 'graduate', 'diploma', 'degree', 
              'degreeyears', 'yearesjob', 'typefull', 'timejob']

docs = preprocess_text(df_jobs, 'description', common_words)

with st.spinner('Training Model ...'):
    n_topics=[3,4,5,6,7,8,9,10]
    
    id2word, corpus = prepare_training_data(docs)
    models, perplexity_scores, coherence_scores = train_models(n_topics, corpus, id2word)

st.session_state.coherence = coherence_scores
st.session_state.perplexity = perplexity_scores
st.session_state.models = models

if 'models' not in st.session_state:
    st.stop()

st.header('Model')
#st.write(type(st.session_state.models).__name__)

#st.header('LDA Ngram')

with st.expander('Model Results'):
    results_df = pd.DataFrame({
        'topic_number': [str(i) for i in n_topics],
        'coherence_score': np.round(coherence_scores, 2),
        'perplexity_score': np.round(perplexity_scores, 2)
    })
    st.dataframe(results_df)

    # Plot Model Results
    plt.rcParams['xtick.labelsize']=5
    plt.rcParams['ytick.labelsize']=5
    fig,ax = plt.subplots(figsize=(3,2))
    ax.plot(n_topics, coherence_scores, color="red", marker=".", markersize=5)
    ax.set_xlabel("Num Topics",fontsize=5)
    ax.set_ylabel("Coherence Score",color="red",fontsize=5)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    ax2.plot(n_topics, perplexity_scores,color="blue",marker=".", markersize=5)
    ax2.set_ylabel("Perplexity Score",color="blue",fontsize=5)

    st.pyplot(fig,use_container_width=False)

    st.markdown('Lower the perplexity better the model. Higher the topic coherence, the topic is more human interpretable.')

# Visualize the topics
if 'models' in st.session_state: 
    with st.form(key='ldavis_form'):
        st.header('Generate Best Model')
        topic_num = st.text_input('Enter Topic Number')
        ldavis_button = st.form_submit_button(label='Generate Model')

    if ldavis_button:
        idx = (np.where(results_df['topic_number']==topic_num)[0][0])        
        lda_model = models[idx]
    
        if lda_model:
            colors = random.sample(COLORS, k=len(n_topics))

            with st.expander('Top N Topic Keywords Wordclouds'):
                topics = lda_model.show_topics(formatted=False, num_words=50,
                                                num_topics=int(topic_num), log=False)
                cols = st.columns(3)
                for index, topic in enumerate(topics):
                    wc = WordCloud(width=700, height=600, background_color='white', 
                                   prefer_horizontal=1.0, color_func=lambda *args, **kwargs: colors[index])
                    with cols[index % 3]:
                        wc.generate_from_frequencies(dict(topic[1]))
                        st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)

            with st.expander('Visualise LDA Topics'):
                with st.spinner('Creating pyLDAvis Visualisation ...'):
                    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
                    vis_html = pyLDAvis.prepared_data_to_html(vis)
                    components.html(vis_html, width=1300, height=800)    

            with st.expander('Topic Word-Weighted Summaries'):
                topic_summaries = {}
                df_topic = pd.DataFrame(columns=['topic','term','weight'])

                for topic in topics:
                    topic_index = topic[0]
                    topic_word_weights = topic[1]
                    topic_summaries[topic_index] = ' + '.join(
                        f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
                    
                    for word, weight in topic_word_weights[:20]:
                        df_new_row = pd.DataFrame({
                            'topic': [topic_index],
                            'term': [word],
                            'weight': [weight*1000]
                        })
                        df_topic = pd.concat([df_topic, df_new_row])
            
            with st.expander('Top Bigram Topic Keywords Wordclouds'):
                bigram_docs = [generate_N_grams(doc,2) for doc in docs]                
                id2word_bigram, corpus_bigram, lda_model_bigram = train_model(bigram_docs, topic_num)

                topics_bigram = lda_model_bigram.show_topics(formatted=False, num_words=50,
                                                      num_topics=int(topic_num), log=False)
                cols = st.columns(3)
                for index, topic in enumerate(topics_bigram):
                    wc = WordCloud(width=700, height=600, background_color='white', 
                                   prefer_horizontal=1.0, color_func=lambda *args, **kwargs: colors[index])
                    with cols[index % 3]:
                        wc.generate_from_frequencies(dict(topic[1]))
                        st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)
            
            with st.expander('Visualise Bigram LDA Topics'):
                with st.spinner('Creating pyLDAvis Visualisation ...'):
                    visbigram = pyLDAvis.gensim.prepare(lda_model_bigram, corpus_bigram, id2word_bigram)
                    visbigram_html = pyLDAvis.prepared_data_to_html(visbigram)
                    components.html(visbigram_html, width=1300, height=800)

            with st.expander('Bigram Topic Word-Weighted Summaries'):
                topic_summaries = {}
                df_bigram_topic = pd.DataFrame(columns=['topic','term','weight'])

                for topic in topics_bigram:
                    topic_index = topic[0]
                    topic_word_weights = topic[1]
                    topic_summaries[topic_index] = ' + '.join(
                        f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
                    
                    for word, weight in topic_word_weights[:20]:
                        df_new_row = pd.DataFrame({
                            'topic': [topic_index],
                            'term': [word.replace('_', ' ')],
                            'weight': [weight*1000]
                        })
                        df_bigram_topic = pd.concat([df_bigram_topic, df_new_row])
                
                for topic_index, topic_summary in topic_summaries.items():
                    st.markdown(f'**Topic {topic_index}**: _{topic_summary}_')
                
                df_bigram_topic.to_csv('results_bigram_topics.csv', index=False)
                st.dataframe(df_bigram_topic)