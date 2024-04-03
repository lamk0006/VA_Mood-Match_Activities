# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:23:11 2024

@author: jlkc1
"""

import numpy as np
import pandas as pd
import os
import re
import faiss
import pygame
import speech_recognition as sr
import geocoder
import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from termcolor import cprint
from gtts import gTTS

pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)

class VirtualAssistant():
    
    def __init__(self):
        self.va_print_without_audio("Running... Say 'wake up' to activate Virtual Assistant or 'turn off' for the Virtual Assistant to sign off")
        self.text = ''
        self.text_emotion = ''
        self.cwd = os.getcwd()
    
    # Function is used to detect emotion based on words. A pre-trained model on emotion is used.
    def emotion_detection(self,sentence):
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
        emotion_res = classifier(sentence)
        self.text_emotion = emotion_res[0]
        return True
    
    # Function is used to ask for the user for input. Incase the input is not correctly captured, the program asks the user up to 3 times what he meant.
    def ask_to_user(self,sentence):
        self.va_print(sentence)
        text_res = self.speech_to_text(param=True)
        count = 0
        while text_res == '$error$':
            if count == 2:
                break
            if count == 1:
                msg_desc = "Sorry, I still don't get what you're saying"
                text_res = self.speech_to_text(param=True,msg_error=msg_desc)
                break
            text_res = self.speech_to_text(param=True)
            count += 1
        self.text = text_res.lower()
        if self.text != '$error$':
            self.user_print(self.text)
    
    # Function is used to load the Indoor/Outdoor activities from a csv file.
    def load_activity_file(self):
        df = pd.read_csv(f"""{self.cwd}/List_Of_Activity.csv""", encoding='latin-1') 
        return df
    
    # Function is used to load the different questions the program can ask to greet the user.
    def load_greet_question(self):
        greet_question = []
        with open(f"""{self.cwd}/Greet_Question.txt""", "r") as f:
            for line in f:
                greet_question.append(line.strip())
        return np.random.choice(greet_question)

    # Function is used to load the different questions the program can ask for the user preferences.
    def load_new_user_question(self):
        new_user_question = []
        with open(f"""{self.cwd}/New_User_Question.txt""", "r") as f:
            for line in f:
                new_user_question.append(line.strip())
        return np.random.choice(new_user_question)
    
    # Function is used to check if the user has any saved preferences.
    def load_user_preference_movie(self):
        return os.stat(f"""{self.cwd}/User_Preference_Movie.txt""").st_size == 0
    
    # Function is used to load the questions to ask the new user.
    def load_mood_question(self,choice):
        if choice == 'movie':
            df_movie = pd.read_excel(f"""{self.cwd}/Mood_Question.xlsx""", sheet_name='Movie_Question') 
            return df_movie

    # Function is used to get/retrieve the user preferences.
    def get_emotion_user_preference(self,emo_state):
        user_preference = []
        with open(f"""{self.cwd}/User_Preference_Movie.txt""", "r") as f:
            for line in f:
                word_list = re.split(r'\t+', line)
                if emo_state in line:
                   user_preference.append(('movie',word_list[1].strip()))                  
        return user_preference       

    # Function is used to ask the user what type of movies per mood he prefers and saves it to a text file.
    def ask_user_preference_movie(self):
        df_movie = self.load_mood_question('movie')
        #Movie
        response_list = []
        for i, row in df_movie.iterrows():
            self.ask_to_user(f"""{row['Question']}""")
            # Remove movie tag
            clean_text = self.text.split()
            new_text = ''
            for i in clean_text:
                if 'movie' in i.lower():
                    continue
                new_text += i + ' '
            response_list.append((row['Mood'],new_text))
        with open(f"""{self.cwd}/User_Preference_Movie.txt""", "w") as f:
            for i in response_list:
                f.write(i[0] + "\t" + i[1] + "\n") 
    
    # Function is used to load the movie dataset.
    def load_movie_dataset(self):
        df_movie_dataset = pd.read_csv(f"""{self.cwd}/preprocessed_movie_dataset.csv""", memory_map=True)
        return df_movie_dataset
    
    # Function is used to create an index of the movie plot based on the user keywords. It uses a pretrained model.
    def movie_semantic_search(self,query_text):
        df = self.load_movie_dataset()
        model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
        encoded_data = model.encode(df['summarization'].tolist())
        encoded_data = np.asarray(encoded_data.astype('float32'))
        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        index.add_with_ids(encoded_data, np.array(range(0, len(df))))
        faiss.write_index(index, 'movie_plot.index')    
        results = self.search(df, query_text, top_k=5, index=index, model=model)
        return results

    # Function is used to retrieve the movie information.
    def fetch_movie_info(self, df, dataframe_idx):
        info = df.iloc[dataframe_idx]
        meta_dict = dict()
        meta_dict['title'] = info['title']
        meta_dict['summarization'] = info['summarization']
        return meta_dict
    
    # Function is used to search for semantic similarity of the keywords entered by the user and the movie plot.
    def search(self, df, query, top_k, index, model):
        query_vector = model.encode([query])
        top_k = index.search(query_vector, top_k)
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results =  [self.fetch_movie_info(df, idx) for idx in top_k_ids]
        return results   
    
    # Function is used to know if the user said Yes or No based on the user input. It uses a pre-trained model.
    def yes_no_question(self,text):
        classifier = pipeline("text-classification", model="manohar899/bert_yes_no", top_k=1)
        res = classifier(text)
        if 'yes' in text:
            return True
        if 'no' in text:
            return False
        if res[0][0]['label'] == 'Yes':
            return True
        return False
    
    # Function is used to detect the sentiment of the user.
    def sentiment_detection(self,sentence):
        classifier = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_res = classifier(sentence)
        return sentiment_res[0]['label'] == 'POSITIVE'
    
    # Function is used to convert speech to text as input.
    def speech_to_text(self,param,msg_error="Sorry, I didn't catch that. Can you please repeat?"):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.2
        with sr.Microphone() as mic:
            try:
                self.va_print_without_audio('Listening...')
                audio = recognizer.listen(mic)
                text = recognizer.recognize_google(audio)
            except:
                if param:
                    self.va_print_failure(msg_error)
                return '$error$'
        return text
    
    # Function is used to print the virtual assistant interactions.
    def va_print(self,text):
        cprint('[Virtual Assistant]', 'white', 'on_cyan', end=" ")
        cprint(text)
        self.audio_play(text)
        print('')

    # Function is used to print the virtual assistant interactions.
    def va_print_failure(self,text):
        cprint('[Virtual Assistant]', 'white', 'on_red', end=" ")
        cprint(text)
        self.audio_play(text)
        print('')

    # Function is used to print the user interactions.
    def user_print(self,text):
        cprint('[User]', 'white', 'on_green', end=" ")
        cprint(text)
        print('')

    # Function is used to print the virtual assistant interactions but without audio.
    def va_print_without_audio(self,text):
        cprint('[Virtual Assistant]', 'white', 'on_magenta', end=" ")
        cprint(text)
        print('')

    # Function is used to convert text to speech for the virtual assistant interactions.
    def audio_play(self,text):
        speaker = gTTS(text=text, lang="en", slow=False)
        path = os.getcwd() + "/speech_t5.wav"
        speaker.save(path)
        # Play the saved audio.
        pygame.mixer.init()
        my_sound = pygame.mixer.Sound(path)
        my_sound.play()
        pygame.time.wait(int(my_sound.get_length() * 1000))

    # Function is used to check if there is no error that was encountered when the user said something.
    def no_error(self,text):
        if text == '$error$' or text == '$skip$' or ('turn off' in text):
            return False
        return True

    # Function is used to retrieve the error that was encounted when the user said something.
    def msg_error(self,text):
        if text == '$error$':
            return "I'm going to sign off. If necessary, you can wake me back up. Thank you."
        if text == '$skip$':
            return ''
        if 'turn off' in text:
            return "Signing off. Thank you"

    # Function is used to get the Virtual Assistant location.
    def get_weather_condition(self):
        myloc = geocoder.ip('me')
        lat = myloc.lat
        lng = myloc.lng
        key = 'ea41b7fca1bc919a190c2a37a6cbcfbe'
        unit = 'metric'
        response = requests.get(f"""https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={key}&units={unit}""")
        if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type',''):
            res_json = response.json()
            location_name = res_json['name']
            temp = round(res_json['main']['temp'])
            weather_prefix = f"""For {location_name}, the temperature is currently {temp} degree celcius."""
            weather_id = res_json['weather'][0]['id']
            weather_desc = ''
            
            if weather_id >= 200 and weather_id < 300:
                weather_desc = "We're expecting a passing thunderstorm today, so brace yourself for some rumbles and heavy rain."
                filter_data = "indoor"
            elif weather_id >= 300 and weather_id < 400:
                weather_desc = "It looks like we'll have drizzle throughout the day."
                filter_data = "indoor"
            elif weather_id >= 500 and weather_id < 600:
                weather_desc = "Expect intermittent rain showers today."
                filter_data = "indoor"
            elif weather_id >= 600 and weather_id < 700:
                weather_desc = "We're in for a snowy day today, with fluffy flakes gently falling from the sky."
                filter_data = "indoor"
            elif weather_id == 800:
                weather_desc = "Today's forecast is a clear sky and abundant sunshine."
                filter_data = "outdoor"
            elif weather_id > 800 and weather_id < 900:
                if weather_id == 801:
                    weather_desc = "Today, there are scattered clouds in the sky, accounting for approximately 11 to 25% coverage."
                if weather_id == 802:
                    weather_desc = "Today, the sky is dotted with scattered clouds, encompassing around 25 to 50% of its coverage."
                if weather_id == 803:
                    weather_desc = "Today, the sky features broken clouds, covering approximately 51 to 84% of its expanse."
                if weather_id == 804:
                    weather_desc = "Today, the sky is heavily overcast, with cloud cover ranging from 85 to 100%."
                filter_data = "outdoor"
            else:
                filter_data = "indoor"
                
        weather_full_desc = ''
        if weather_desc != '':
            weather_full_desc = weather_prefix + ' ' + weather_desc
        else:
            weather_full_desc = weather_prefix
        return weather_full_desc, filter_data
    
    # Function is used to recommend an activity based on the weather.
    def recommend_an_activity(self,emo_state):
        df = self.load_activity_file()
        # Based on the weather, propose an indoor or outdoor activity
        weather_condition, filter_data = self.get_weather_condition()
        self.va_print(weather_condition)
        df = df.loc[df['Mood'] == emo_state]
        df = df.loc[df['Indoor/Outdoor'] == filter_data]
        reindex_df = df.reindex(np.random.permutation(df.index)).head(5)
        count_activity = 0
        # The user is recommended randomly 2 activities based on his mood/emotion.
        for index, row in reindex_df.iterrows():
            if count_activity > 1:
                break
            if count_activity == 0:
                self.va_print(f"""I've come across an {row['Indoor/Outdoor']} activity that might catch your interest. It entails {row['Activities']}.""")
            if count_activity == 1:
                self.va_print(f"""I've found a different activity, and it is {row['Activities']}""")
            ques = 'What do you think about this activity?'
            self.ask_to_user(f"""{ques}""")
            if not self.no_error(va.text):
                self.va_print(va.msg_error(va.text))
                break
            if not self.sentiment_detection(self.text):
                if count_activity == 0:
                    self.va_print(f"""I'm sorry that this wasn't what you were looking for. I am searching for another {row['Indoor/Outdoor']} activity for you.""")
                else:
                    self.va_print("I'm sorry that this wasn't what you were looking for. I do not have any other activity to recommend.")
                    self.va_print("Feel free to wake me back up if you want any other activity. I am signing off. Thank you.")
                    break
            else:
                self.va_print("That's fantastic! I'm glad to hear that.")
                self.va_print("I am going to sign off. If necessary, you can wake me back up. Thank you.")
                break
            count_activity = count_activity + 1
 
if __name__ == "__main__":
    
    # Create a new instance of the class VirtualAssistant defined above.
    va = VirtualAssistant()

    run_wake_word = True
    run = True
    
    # Loop until the wake up word is said by the user.
    while run_wake_word:
        text = va.speech_to_text(param=False)
        if va.no_error(text): 
            va.user_print(text)
        else:
            va.user_print('')
        if not ("wake up" in text):
            va.text = ''
            continue
        else:
            va.text = text
            run = True
        
        # Once the wake up word is detected, the Virtual Assistant can start interacting with the user.
        while run:
            # Wake up the virtual assistant on specific keywords
            if 'wake up' in va.text:
                if va.load_user_preference_movie():
                    ques = va.load_new_user_question()
                    va.ask_to_user(f"""{ques}""")
                    if not va.no_error(va.text):
                        va.va_print(va.msg_error(va.text))
                        run = False
                        break
                    if va.yes_no_question(va.text):
                        va.ask_user_preference_movie()
                        va.va_print('Your preferences have been saved. Thank you!')
                    ques = va.load_greet_question()
                    va.ask_to_user(f"""{ques}""")
                    if not va.no_error(va.text):
                        va.va_print(va.msg_error(va.text))
                        run = False
                        break
                else:
                    ques = va.load_greet_question()
                    va.ask_to_user(f"""{ques}""")
                    if not va.no_error(va.text):
                        va.va_print(va.msg_error(va.text))
                        run = False
                        break
            elif va.emotion_detection(va.text):
                emo_state = va.text_emotion[0]['label']
                # Define questions based on the mood of the user.
                if emo_state == "anger":
                    ques = "Hey, It seems like you're feeling upset. I'm here for you. Do you want to watch a movie or particiapte in another activity?"
                elif emo_state == "fear":
                    ques = "Oh no, something's scaring you. It's okay, I'm here. Want to watch a movie or do another activity to help you feel better?"
                elif emo_state == "joy":
                    ques = "Hey, you seem happy! It's great to see. Let's keep it going. Can I recommend you a movie to watch or another activity?"
                elif emo_state == "sadness":
                    ques = "It seems like you're feeling down. I'm here if you need me. Can I recommend you a movie to watch or another activity to lift your spirit?"
                else:
                    if not (emo_state in ['anger','fear', 'joy', 'sadness']):
                        va.va_print("Apologies, but I can't identify your current mood. It is out of my scope.")
                        va.va_print("I'm signing off. If necessary, you can wake me again. Thanks!")
                        run = False
                        break
                va.ask_to_user(f"""{ques}""")
                if not va.no_error(va.text):
                    va.va_print(va.msg_error(va.text))
                    run = False
                    break
                if va.yes_no_question(va.text):
                    # Recommend a movie
                    if 'movie' in va.text:
                        # Ask the user if he wants a movie based on his preferences or if he wants to look for another movie based on his mood.
                        if not va.load_user_preference_movie():
                            ques = 'Do you want a movie based on your user preference?'
                            va.ask_to_user(f"""{ques}""")
                        else:
                            va.text = '$skip$'
                        to_query = ''
                        # Recommend a movie based on the user preferences. The mood affects the search of the movie.
                        user_pref_op = False
                        if va.yes_no_question(va.text) and va.no_error(va.text):
                            user_pref_op = True
                            user_pref = va.get_emotion_user_preference(emo_state)
                            to_query = user_pref[0][1]
                        else:
                            # Dynamic search for a movie by user keywords.
                            movie_item = ''
                            if emo_state == "anger":
                                movie_item = "Relaxing comedies, Positive vibe movies, Happy movies??"
                            elif emo_state == "fear":
                                movie_item = "Comforting movies, Relaxing comedies, Funny movies?"
                            elif emo_state == "joy":
                                movie_item = "Comedy movies, Romantic drama movies, Adventure movies?"
                            elif emo_state == "sadness":
                                movie_item = "Fantasy movies, Relaxing comedy movies, Romantic movies"
                            ques = f"""What type of movie are you in the mood to watch? {movie_item}"""
                            va.ask_to_user(f"""{ques}""")
                            split_text = va.text.split()
                            to_query = ''
                            # Remove the keyword movie from the string as it will affect the semantic search.
                            for i in split_text:
                                if 'movie' in i.lower():
                                    continue
                                to_query += i + ' '
                        if not va.no_error(va.text):
                            va.va_print(va.msg_error(va.text))
                            run = False
                            break
                        va.va_print_without_audio('Words used for semantic search: ' + to_query)
                        results = va.movie_semantic_search(to_query)
                        count_movie = 0
                        # The top 5 movies that has similar semantic is queried and the user is recommended the top 2 based on the semantic score.
                        for i in results:
                            if count_movie > 1:
                                run = False
                                break
                            if count_movie == 0:
                                if user_pref_op == True:
                                    va.va_print(f"""Based on your preferences, I found a movie that might interest you. The movie name is {i['title']}""")
                                else:
                                    va.va_print(f"""Based on the keywords you said, I found a movie that might interest you. The movie name is {i['title']}""")
                            if count_movie == 1:
                                va.va_print(f"""I have found another movie. The movie name is {i['title']}""")
                            # Ask the user if he would like to have a summary of the movie.
                            ques = 'Would you like me to give you a brief summary of the movie?'
                            va.ask_to_user(f"""{ques}""")
                            if not va.no_error(va.text):
                                va.va_print(va.msg_error(va.text))
                                run = False
                                break
                            if va.yes_no_question(va.text):
                                va.va_print(f"""Here's a quick overview of the movie {i['title']}""")
                                va.va_print(f"""{i['summarization']}""")
                                ques = 'What did you think of the movie?'
                                va.ask_to_user(f"""{ques}""")
                                if not va.no_error(va.text):
                                    va.va_print(va.msg_error(va.text))
                                    run = False
                                    break
                                # Get the sentiment from the user if he likes the movie based on the summary and if not, the VA recommends the user another movie.
                                if not va.sentiment_detection(va.text):
                                    if count_movie == 0:
                                        if user_pref_op:
                                            va.va_print("I'm sorry the movie is not what you were hoping for. I am searching for another movie based on your preferences.")
                                        else:
                                            va.va_print("I'm sorry the movie is not what you were hoping for. I am searching for another movie based on the information you gave me.")
                                    else:
                                        va.va_print("I'm sorry the movie is not what you were hoping for.")
                                        ques = "Do you want me to recommend you another activity instead?"
                                        va.ask_to_user(f"""{ques}""")
                                        if va.yes_no_question(va.text):
                                            va.recommend_an_activity(emo_state)
                                        run = False
                                        break
                                else:
                                    va.va_print("That's fantastic! I'm glad to hear that.")
                                    va.va_print("Feel free to reach out if you need any other movie. I am signing off. If necessary, you can wake me again. Thank you.")
                                    run = False
                                    break
                            else:
                                ques = "Would you like me to recommend another movie?"
                                va.ask_to_user(f"""{ques}""")
                                if not va.no_error(va.text):
                                    va.va_print(va.msg_error(va.text))
                                    run = False
                                    break
                                if not va.yes_no_question(va.text):
                                    va.va_print("I regret that I couldn't find a movie that matches your current mood.")
                                    ques = "Do you want me to recommend you another activity instead?"
                                    va.ask_to_user(f"""{ques}""")
                                    if va.yes_no_question(va.text):
                                        va.recommend_an_activity(emo_state)
                                    run = False
                                    break
                            count_movie = count_movie + 1  
                    elif 'activity' in va.text:
                        # Recommend an activity
                        va.recommend_an_activity(emo_state)
                        run = False
                        break
                    else:
                        # Cannot identify whether the user wants a movie or an activity
                        va.va_print("I'm sorry to say that I couldn't determine if you're interested in a movie or an activity.")
                        va.va_print("I'm signing off for now. If necessary, you can wake me again. Thanks!")
                        run = False
                        break
                else:
                    # Cannot identify whether the user said yes or no to a movie or activity
                    va.va_print("I regret to inform you that I couldn't determine which option you want.")
                    va.va_print("I'm signing off for now. If necessary, you can wake me again. Thanks!")
                    run = False
                    break