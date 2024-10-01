import time

import requests
import pandas as pd

BASE_URL = 'https://api.jikan.moe/v4/'
def fetch_data(start, nb):
    df = pd.DataFrame(columns=['title', 'episodes', 'rating', 'studio', 'genre1', 'genre2', 'synopsis'])
    request_per_second = 3
    request_per_minute = 60
    nb_request_per_second = 0
    nb_request_per_minute = 0
    nb_succeful = 0
    i = start
    last_sucessful_episode = i
    while nb_succeful < nb + 1 and i < last_sucessful_episode + 100:
        url = BASE_URL + 'anime/' + str(i) + '/full'
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            titles = data['data']['titles']
            episodes = data['data']['episodes']
            rating = data['data']['rating']
            synopsis = data['data']['synopsis']
            if len(data['data']['producers']) > 0:
                producers = data['data']['producers'][0]['name']
            else:
                producers = None
            if len(data['data']['genres']) > 0:
                genre1 = data['data']['genres'][0]['name']
            else:
                genre1 = None
            if len(data['data']['genres']) > 1:
                genre2 = data['data']['genres'][1]['name']
            else:
                genre2 = None

            for t in titles:
                if t['type'] == 'Default':
                    title = t['title']

            df1 = pd.DataFrame({'title': [title], 'episodes': [episodes], 'rating': [rating], 'studio': [producers], 'synopsis': [synopsis], 'genre1': [genre1], 'genre2': [genre2]})
            df = pd.concat([df, df1], axis=0)

            nb_request_per_minute += 1
            nb_request_per_second += 1
            nb_succeful += 1
            last_sucessful_episode = i

            print(url + "---" + str(nb_succeful))

            if nb_request_per_second >= request_per_second:
                nb_request_per_second = 0
                time.sleep(1)

            if nb_request_per_minute >= request_per_minute:
                nb_request_per_minute = 0
                time.sleep(60)
        i+=1

    df.to_csv('data/data.csv', index=False, mode='a', header=False)

fetch_data(18724, 500)