import os
import re
import sys                                                      import argparse
import joblib
import logging
import requests
import threading
import queue
import asyncio
import aiohttp
import json
import csv
import pandas as pd
import numpy as np                                              import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression             from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline                           from urllib.parse import urljoin, urlparse
from typing import List, Dict, Tuple, Optional, Any
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting Application")

class Config:
    seed_url = "https://oxfordlearnersdictionaries.com/definition/english/"
    depth = 2
    max_threads = 10
    max_workers = 5
    semaphore = 10
    max_links = 3000
    max_rows = 6
    columns = ["word", "meaning", "source"]
    output_file = "data.json"
    output_format = "json"
    words_file = "all.txt"
    model_name = "model.joblib"
    save_model = True
    mode = "async"
    mode_process = "bs4htmlparser"
    mode_chart = "plot"
    remove_tags = ["script", "style"]
    timeout = 5
    runner = "async"

def load_words():
    with open(Config.words_file, "r") as r:
        content = r.read().splitlines()
    return content

class Workers:
    def __init__(self, mode):
        self.mode = mode

    # Alternative Method: Through Api
    @staticmethod
    def get_definition(word):
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        resp = requests.get(url)
        logging.info(f"Scraped: {url}")
        if resp.status_code == 200:
            data = resp.json()
            definitions = []
            for meaning in data[0]['meanings']:
                for definition in meaning['definitions']:
                   definitions.append(definition['definition'])

            wordd = word
            definition = definitions[0] # First definition only

            return wordd, definition, url
        return word, None, url

    class AsyncWorker:
        def __init__(self, timeout=Config.timeout, semaphore=Config.semaphore, max_links=Config.max_links):
            self.timeout = timeout
            self.semaphore = asyncio.Semaphore(semaphore)
            self.max_links = max_links


        async def scrape(self, session, url):
            try:
                async with asyncio.timeout(Config.timeout):
                    async with self.semaphore:
                        async with session.get(url) as response:
                            res = await response.text()

                            fixed_data = {
                                    "url": url,
                                    "content": f"{res}"
                            }

                            return fixed_data
            except TimeoutError:
                return f"Failed {url}"

    class ThreadedWorker:
        def __init__(self, timeout=Config.timeout, words_file=Config.words_file, max_links=Config.max_links):
            self.timeout = timeout
            self.words_file = words_file
            self.max_links = max_links

        def scrape(self, url):
            try:
                r = requests.get(url, timeout=Config.timeout)

                fixed_data = {
                    "url": url,
                    "content": f"{r.text}"
                }

                return  fixed_data
            except TimeoutError:
                return f"Failed {url}"

# For manual parsing
class ProcessData:
    def __init__(self, mode):
        self.mode = mode

    class Bs4Parser:
        def __init__(self, dict_data):
            self.dict_data = dict_data
            self.parser_type = parser_type

        def parse(self):
            soup = BeautifulSoup(html_content, 'html.parser')
            definition_span = soup.find('span', class_='def')
            definition = definition_span.text.strip()
            return definition

    class RegexParser:
        def __init__(self, dict_data):
            self.dict_data = dict_data

        def parser(self):
            pattern = r'<span class="def"[^>]*>(.*?)</span>'
            matches = re.findall(pattern, html_content)

            definition =  matches[0]

            return definition


class ProcessnClassify:
    def __init__(self, df_dict_data, columns=Config.columns, output_file=Config.output_file, output_format=Config.output_format):
        self.df_dict_data = df_dict_data
        self.columns = columns
        self.output_file = output_file
        self.output_format = output_format

    @staticmethod
    def Apiprocessor(data):
        df = pd.DataFrame(data, columns=Config.columns)

        if Config.output_format == "json":
            df.to_json(Config.output_file)
        else:
            df.to_csv(Config.output_file)

        logging.info(f"Processed and saved {len(df)} rows")

    def processor(self, df):
        df = pd.DataFrame(self.df_dict_data, columns=self.columns)
        logging.info("Prepared the Dataset")

        # To allow chaining
        return self

    def store(self, df):
        if Config.output_format == "json":
            df.to_json(self.output_file)
            logging.info(f"Saved to {Config.output_format}")
        elif Config.output_format == "csv":
            df.to_csv(self.output_file)
            logging.info(f"Saved to {Config.output_format}")
        elif Config.output_format == "excel":
            df.to_excel(self.output_file)
            logging.info(f"Saved to {Config.output_format}")
        else:
            logging.error(f"Cannot save with  this format {Config.output_format}")

        return self

    # Pre-Train to test data quality
    def classify_data(self, df):
        """WORKFLOW:
        Classifies the scraped data and tests its accuracy
        """
        X_train, X_test, Y_train, Y_test = train_test_split(
            df["word"],
            df["meaning"],
            test_size=0.2,
            random_state=42
        )

        model = Pipeline([
            ("vectorizer", TfidfVectorizer(ngram_range=(1,3), stop_words="english")),
            ("clf", LogisticRegression(max_iter=500))
        ])

        model.fit(X_train, Y_train)
        preds = model.predict(X_test)

        print(f"accuracy: {accuracy_score(Y_test, preds)}")
        print(classification_report(Y_test, preds))

        # Run some tests
        logging.info("testing the dataset...")
        prediction1 = model.predict(["sophisticated"])[0]
        prediction2 = model.predict(["aisle"])[0]
        prediction3 = model.predict(["grammer"])[0]

        print(f"Q: sophisticated\nA: {prediction1}")
        print(f"Q: aisle\nA: {prediction2}")
        print(f"Q: grammer\nA: {prediction3}")

        if Config.save_model == True:
           joblib.dump(model, Config.model_name)
           logging.info("Saved the model")
        else:
            logging.info("Done classifying")

class Runner:
    def __init__(self, runner=Config.runner):
        self.runner = runner

    async def Asyncrunner(self):
        seed_url = Config.seed_url
        words = load_words()

        full_domain = []
        for word in words:
            full_url = seed_url + word + f"?q={word}"
            full_domain.append(full_url)

            async with aiohttp.ClientSession() as session:
                tasks = [Workers.AsyncWorker().scrape(session, url) for url in full_domain]
                results = await asyncio.gather(*tasks)
                return results

    def Threadingrunner(self):
        seed_url = Config.seed_url
        words = load_words()

        full_domain = []
        for word in words:
            full_url = seed_url + word + f"?q={word}"
            full_domain.append(full_url)

        tasks = []

        for url in full_domain:
            target = Workers.ThreadedWorker()
            t = threading.Thread(target=target.scrape, args=(url,))
            t.start()
            tasks.append(t)

        for t in tasks:
            t.join()

    @staticmethod
    def ApiRunner():
        words = load_words()

        data = []
        for word in words:
            res = Workers.get_definition(word)
            data.append(res)

        logging.info(f"Scrapped {len(data)} words")
        logging.info("Processing...")
        ProcessnClassify.Apiprocessor(data)

  """
  UNDER PRODUCTION
  """
