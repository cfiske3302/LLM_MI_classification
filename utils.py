import json
import numpy
import csv
import openai
import time
import os
from dotenv import load_dotenv
import codecs
import multiprocessing
load_dotenv('.env')
from prompts import create_llm_classification_prompt, MISC_DICTS
from utils import *
from tqdm import tqdm

def pretty_print(path):
    with open(path, 'r') as f:
        for line in f:
            js = json.loads(line)
            message = js['interlocutor']
            if message == "therapist":
                message += f" ({js['llm_classification']})"
            message += f": {js['utterance_text']}"
            print(message)

def read_dict_list(path):
    dict_list = []
    with open(path, 'r') as f:
        for line in f:
            js = json.loads(line)
            dict_list.append(js)
    return dict_list

def get_completion_from_messages(messages, temperature=0):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    model = 'gpt4o'
    # model = 'gpt-4'
    for i in range(3):
        try:
            response = openai.ChatCompletion.create(
                                            model=model,
                                            messages=messages,
                                            temperature=temperature, 
                                        )
            return response.choices[0].message["content"]
        except Exception as e:
            if i == 2:
                print(e)
                return 'error'
            time.sleep(3*(2**i))