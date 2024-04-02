# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:47:46 2024

@author: jlkc1
"""

from transformers import pipeline
import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

movie_plot_treshold = 100
cwd = os.getcwd()

def summarization_movie_plot(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    movie_plot = f"""{text}"""
    summary = summarizer(movie_plot, max_length=movie_plot_treshold, do_sample=False)
    return summary[0]['summary_text']

def load_movie_dataset():
    df_movie_dataset = pd.read_csv(f"""{cwd}/tmdb_5000_movies.csv""", memory_map=True)
    df_movie_dataset.dropna(subset=['overview'],inplace=True)
    df_movie_dataset.drop_duplicates(subset=['overview'],inplace=True)
    df_movie_dataset['Length of overview'] = df_movie_dataset['overview'].apply(lambda words: len(words.split()))
    df_movie_dataset = df_movie_dataset[df_movie_dataset['Length of overview'] > 0]
    
    # Summarization
    result = []
    for index, row in df_movie_dataset.iterrows():
        if row['Length of overview'] > movie_plot_treshold:
            result.append(summarization_movie_plot(row['overview']))
        else:
            result.append(row['overview'])
            
    df_movie_dataset['summarization'] = result
    df_movie_dataset['Length of summarization'] = df_movie_dataset['summarization'].apply(lambda words: len(words.split()))

    return df_movie_dataset

def export_movie_dataset(df):
    df.to_csv(f"""{cwd}/preprocessed_movie_dataset.csv""", encoding='utf-8', index=False)

if __name__ == "__main__":
    df = load_movie_dataset()
    export_movie_dataset(df)