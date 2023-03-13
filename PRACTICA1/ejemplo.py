import pandas as pd
import numpy as np
import re


def separate_tweets(file_name):

    # Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
    df = pd.read_csv(file_name, sep='\t', dtype=str)
    tweets = df.drop(columns=['ID','anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust'])
    # tweets = df.to_string()
    tweets = tweets.to_string(index=False)
    
    #tweets = np.array2string(tweets,threshold=3999)
    return tweets


if __name__ == "__main__":

    tweets = separate_tweets('tweets.txt')
    
    #hashtags
    hashtags = re.compile("\W#\w+")
    hashtagsFound = re.findall(hashtags, tweets)
    print("Hashtags Found:")
    print(len(hashtagsFound))
    
    #usernames
    usernames = re.compile("[\W\s]@[\w+]{1,15}")
    usernamesFound = re.findall(usernames, tweets)
    print("Usernames Found:")
    print(len(usernamesFound))
    
    #times
    times = re.compile("(([0-2]\d)|\D\d){1}:([0-5]\d){1}")
    timesFound = re.findall(times,tweets)
    print("Times Found:")
    print(len(timesFound))
    
    #dates
    dates1 = re.compile("([0-3]\d/[0-1]\d/[0-2]\d\d\d)")
    dates1Found = re.findall(dates1,tweets)
    dates2 = re.compile("([0-2]\d\d\d/[0-1]\d/[0-3]\d)")
    dates2Found = re.findall(dates2,tweets)
    dates3 = re.compile("([1-3]?\d/1?\d/[1-2]\d\d\d)")
    dates3Found = re.findall(dates3,tweets)
    dates4 = re.compile("([1-2]\d\d\d/1?\d/[1-3]?\d)")
    dates4Found = re.findall(dates4,tweets)
    dates5 = re.compile("(1?\d/[1-3]?\d)")
    dates5Found = re.findall(dates5,tweets)
    dates6 = re.compile("([1-3]\d/1?\d)")
    dates6Found = re.findall(dates6,tweets)
    dates7 = re.compile("((Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)/[1-3]?\d)")
    dates7Found = re.findall(dates7,tweets)
    dates8 = re.compile("([1-3]\d/(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre))")
    dates8Found = re.findall(dates8,tweets)
    dates9 = re.compile("([0-2]\d\d\d/(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)/[1-3]?\d)")
    dates9Found = re.findall(dates9,tweets)
    dates10 = re.compile("([1-3]\d/(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)/[0-2]\d\d\d)")
    dates10Found = re.findall(dates10,tweets)
    dates11 = re.compile("([0-2]\d\d\d/(Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)/[1-3]?\d)")
    dates11Found = re.findall(dates11,tweets)
    dates12 = re.compile("([1-3]\d/(Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic)/[0-2]\d\d\d)")
    dates12Found = re.findall(dates12,tweets)
    print("Dates Found:")
    print(len(dates1Found)+len(dates2Found)+len(dates3Found)+len(dates4Found)+len(dates5Found)+len(dates6Found)+len(dates7Found)+len(dates8Found)+len(dates9Found)+len(dates10Found)+len(dates11Found)+len(dates12Found))
    
    #emoticonos
    emoticonos = re.compile("(:|;|x|X)(\)|\(|/|p|\||D|v)")
    emoticonosFound = re.findall(emoticonos,tweets)
    print("Emoticonos Found:")
    print(len(emoticonosFound))
    
    