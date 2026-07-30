import os
import re
import sys                                                    
import argparse
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
import numpy as np                                          
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression            
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline                          
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Tuple, Optional, Any
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("[+]Starting Application")

class Config:
    seed_url = "https://oxfordlearnersdictionaries.com/definition/english/"
    depth = 2
    max_threads = 100
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
    return content[:1000]

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
                            print(f"Visited {url}")

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
            self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"}

        def scrape(self, url):
            try:
                r = requests.get(url, timeout=Config.timeout, headers=self.headers)
                print(f"Scraped {url}")

                fixed_data = {
                    "url": url,
                    "content": f"{r.text}"
                }

                return  fixed_data
            except requests.exceptions.Timeout:
                return f"Failed {url}"
            except requests.RequestException as e:
                return {"url": url, "content": None, "error": str(e)}

# For manual parsing
class ProcessData:
    def __init__(self, mode):
        self.mode = mode

    class Bs4Parser:
        def __init__(self, data):
            self.data = data

        def parse(self):
            soup = BeautifulSoup(self.data, 'html.parser')
            # Primary selector used on Cambridge entries
            definition_tag = soup.select_one("div.def.ddef_d.db")
            if definition_tag:
                # preserve spaces between nested tags
                return definition_tag.get_text(" ", strip=True)

            # fallback: some pages put the definition in a span.def
            definition_span = soup.find('span', class_='def')
            if definition_span:
                return definition_span.get_text(" ", strip=True)

            # fallback: meta description
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.has_attr("content"):
                return meta["content"].strip()

            return None

    class RegexParser:
        def __init__(self, data):
            self.data = data

        def parse(self):
            # Try to capture the Cambridge definition block or a span.def
            pattern = r'<div[^>]*class=[\"\'][^\"\']*def[^\"\']*ddef_d[^\"\']*db[^\"\']*[\"\'][^>]*>(.*?)</div>'
            matches = re.findall(pattern, self.data, flags=re.S)
            if matches:
                text = re.sub(r'<[^>]+>', ' ', matches[0])
                return re.sub(r'\s+', ' ', text).strip()

            # fallback to span.def
            pattern2 = r'<span[^>]*class=[\"\'][^\"\']*def[^\"\']*[\"\'][^>]*>(.*?)</span>'
            matches2 = re.findall(pattern2, self.data, flags=re.S)
            if matches2:
                text = re.sub(r'<[^>]+>', ' ', matches2[0])
                return re.sub(r'\s+', ' ', text).strip()

            return None


class ProcessnClassify:
    def __init__(self):
        pass

    @staticmethod
    def Apiprocessor(data):
        df = pd.DataFrame(data, columns=Config.columns)

        if Config.output_format == "json":
            df.to_json(Config.output_file)
        else:
            df.to_csv(Config.output_file)

        logging.info(f"Processed and saved {len(df)} rows")

    def processor(self, dict_data):
        df = pd.DataFrame(dict_data, columns=Config.columns)
        logging.info("Prepared the Dataset")
        logging.info(f"Dataset: {df.to_string}")
        df.to_json("data.json", index=False)

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

    async def Asyncrunner():
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

    def ThreadingRunner(self):
        words = load_words()
        full_domain = [Config.seed_url + w + f"?q={w}" for w in words]
                    
        results = []
        with ThreadPoolExecutor(max_workers=Config.max_threads) as executor:
            futures = [executor.submit(Workers.ThreadedWorker().scrape, url) for url in full_domain]
            for future in as_completed(futures):
                results.append(future.result()) # <-- collects return value
                                                            
        return results

    @staticmethod
    def ApiRunner():
        words = load_words()

        data = []
        for word in words:
            res = Workers.get_definition(word)
            data.append(res)

        logging.info(f"Scrapped {len(data)} words")

if __name__ == "__main__":
    # Initialize main workflow
    logging.info("[+]Scraping Data...")
    data = Runner().ThreadingRunner()
    logging.info("[+]Finished Scraping...")

    dict_data = []
    for d in data:
        # d may be a dict from ThreadedWorker or an error string
        content = None
        url = None
        if isinstance(d, dict):
            content = d.get("content")
            url = d.get("url")

        parsed = ProcessData.Bs4Parser(content).parse() if content else None

        # extract word from url query param ?q=word
        word = None
        if url:
            match = re.search(r'[?&]q=([^&]+)', url)
            if match:
                word = match.group(1)

        dict_data.append((word, parsed, url))

    logging.info("[+]Processing Data into a dataset")
    ProcessnClassify().processor(dict_data)

    # Load saved data for classification
    try:
        df = pd.read_json("data.json")
        ProcessnClassify().classify_data(df)
    except Exception as e:
        logging.error(f"Failed to load/process data.json: {e}")
