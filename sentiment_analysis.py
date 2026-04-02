import pandas as pd
import numpy as np
import re
import contractions
import nltk
import os
import torch
import warnings
import logging

# Silence library warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from nltk.corpus import reuters, stopwords
from scipy.special import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/reuters')
except LookupError:
    nltk.download('reuters')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize RoBERTa globally
MODEL_URL = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_URL)
# Direct memory processing module
def date_preprocess(data):
  # Convert to datetime type, invalid ones → NaT (Not a Time)
  data['date'] = pd.to_datetime(data['date'], errors='coerce')
  # Remove rows with invalid date, ie, removing NaT values
  data = data.dropna(subset=['date'])
  return data


CRYPTO_HASHTAG_VOCAB = {
    # Coins / assets
    "bitcoin", "btc", "ethereum", "eth", "bnb", "solana", "sol",
    "xrp", "dogecoin", "doge", "cardano", "ada",

    # Market sentiment
    "bullish", "bearish", "bullrun", "bearmarket",
    "moon", "to_the_moon", "pump", "dump",
    "breakout", "breakdown", "rally", "crash",

    # Price & metrics
    "ath", "atl", "high", "low",
    "resistance", "support",
    "volume", "volatility", "dominance",

    # Trading actions
    "buy", "sell", "hold", "hodl",
    "long", "short", "leverage",
    "entry", "exit", "takeprofit", "stoploss",

    # News & events
    "etf", "halving", "airdrop",
    "listing", "delisting",
    "regulation", "sec", "fed",

    # Slang / emotions
    "fomo", "fud", "rekt", "ngmi", "wagmi",
    "lfg", "gm", "gn"
}

# download necessary hastag words (reuters) from nltk
nltk.download('reuters')

# Build Reuters English vocabulary which will use to expand words
REUTERS_VOCAB = set(word.lower() for word in reuters.words())
# Addin some crypto related hashtag wors to the REUTERS_VOCAB
REUTERS_VOCAB.update(CRYPTO_HASHTAG_VOCAB)

CRYPTO_SLANG_ACRONYM_MAP = {
    # Slang terms
    "hodl": "hold",
    "fomo": "fear of missing out",
    "fud": "fear uncertainty doubt",
    "rekt": "financially ruined",
    "moon": "price increase rapidly",
    "dump": "sharp price fall",
    "pump": "artificial price increase",
    "bagholder": "investor holding losses",
    "whale": "large investor",
    "bearish": "expecting price fall",
    "bullish": "expecting price rise",
    "ath": "all time high",
    "atl": "all time low",

    # Acronyms
    "btd": "buy the dip",
    "dyor": "do your own research",
    "ngmi": "not going to make it",
    "wagmi": "we are going to make it",
    "lmao": "laughing my ass off",
    "imo": "in my opinion",
    "idk": "i do not know"
}

# download stopword sets from nltk
nltk.download('stopwords')

# stop words are words that less or no contribute to the sentiments such as the, to, an, and, etc
STOP_WORDS = set(stopwords.words('english'))
# Preserve negations which has a large contributions to sentiments
NEGATIONS = {"not", "no", "nor", "never", "don't", "dont", "can't", "won't", "n't"}
STOP_WORDS = STOP_WORDS - NEGATIONS


# text or tweets are preprocessing using regular expressiond (re -> python library)
def text_preprocessing(text):
  # removing retweets like: "RT @user: text"
  text = re.sub(r'(?i)^\s*rt\s+@?\w+:?', '', text)

  # first removing the urls from the tweets
  text = re.sub(r'https?://\S+|www\.\S+', '', text) # \S -> all characters including nums and special symbls until a space or newline chars

  # removing the mentions, (?<!\w)@ -> checking the char immediately before @ is a space not a letter, num or any other char
  text = re.sub(r'(?<!\w)@[a-zA-Z0-9_]+', '', text)

  # removing extra spaces and newline characters into a single space
  text = re.sub(r'\s+', ' ', text)
  text = text.strip() # removing leading and trailing spaces

  # shorten repeated characters, nooooo -> noo
  text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

  # lowercasing all the characters in the text excluding hashtags
  text = text.lower()

  # removing short tweets, length of words < 4
  if len(text.split()) < 4:
    return None

  # Process hashtags using Reuters dictionary
  def handle_hashtag(match):
      word = match.group(1)
      if word in REUTERS_VOCAB:
          return word # removing the '#' symbols, keeping the words only
      else:
          return '' # remove entire hashtag
  text = re.sub(r'#([a-z]+)', handle_hashtag, text) # calling the handle_hastag function where each hastag as parameter implicitly passed

  # contractions is a library contains dictionary of English contractions, eg: "you're" -> "you are", "they've" -> "they have"
  def expand_contractions(text):
    return contractions.fix(text)
  text = expand_contractions(text) # expanding the short forms using expand_contractions function

  # expanding the slang or acronyms to their expanded forms, fomo -> fear of missing out, using the CRYPTO_SLANG_ACRONYM_MAP created above
  def expand_crypto_slang_acronyms(text):
    for term, expand in CRYPTO_SLANG_ACRONYM_MAP.items():
      pattern = rf'\b{term}\b' # r -> raw string using in regex that not treating \b as blackslashes instead using it has regex boundary (\b...\b), only take single 'ath' not the 'ath' in word 'bath'
                               # f -> formated string used to replace the 'term' with appropriate word
      text = re.sub(pattern, expand, text)
    return text
  text = expand_crypto_slang_acronyms(text)

  # removing the ticker symbols such as $btc, $ETH, etc
  text = re.sub(r'\$[A-Za-z]{2,}', '', text)
  # removing the '$' for next step
  text = re.sub(r'\$', '', text)

  # removing the words with nums
  text = re.sub(r'\b\w*\d+\w*\b', '', text, flags=re.UNICODE)

  # remove punctuation and excessive special characters
  text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+", " ", text)
  text = re.sub(r"\s+", " ", text).strip()

  # removing the stop words
  def removing_stopwords(text):
    text = text.split() # splitting list of tokens -> ['bitcoin', 'is', 'high']
    text = [token for token in text if token not in STOP_WORDS] # only take the word that is not a stop word
    return ' '.join(text)
  text = removing_stopwords(text)

  return text


# Minimal cleaning for non-English tweets
def minimal_text_preprocess(text):
  text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove urls
  text = re.sub(r'@\w+', '', text) # remove mentions
  text= re.sub(r'\s+', ' ', text).strip() # normalize spaces
  text = text if len(text.split()) >= 3 else None # take the texts with minimum 3 words in it
  return text
from fast_langdetect import detect as fast_detect
from tqdm import tqdm
tqdm.pandas()


def lang_detect(text):
    try:
        if not isinstance(text, str) or len(text.strip()) < 10:
            return "unknown"
        # fast_detect returns a list of dictionaries like [{'lang': 'en', 'score': 0.8}]
        results = fast_detect(text)
        if results and results[0]['score'] > 0.5:
            return results[0]['lang']
        return "unknown"
    except Exception:
        return "err"


def preprocess_multilang_tweets(chunk, col='text'):
    processed_texts = []
    for text in tqdm(chunk[col], total=len(chunk), desc="Preprocessing tweets", leave=False):
        if not isinstance(text, str):
            processed_texts.append(None)
            continue
            
        lang = lang_detect(text)
        
        if lang == "en" or lang == "unknown":
            processed_texts.append(text_preprocessing(text))
        elif lang == "err":
            processed_texts.append(None)
        else:
            processed_texts.append(minimal_text_preprocess(text))
            
    chunk['processed_text'] = processed_texts
    return chunk


# Crypto-specific sentiment terms
CRYPTO_LEXICON = {
    "bullish": 3.2,
    "bearish": -3.1,
    "hodl": 2.4,
    "fomo": -1.9,
    "ath": 3.4,
    "moon": 3.5,
    "pump": 2.8,
    "dump": -2.9,
    "rekt": -3.6,
    "rugpull": -3.9,
    "scam": -4.0,
    "crash": -3.5,
    "breakout": 3.0,
    "accumulate": 2.1,
    "selloff": -2.6,
}

# Loughran–McDonald inspired financial terms
FINANCIAL_LEXICON = {
    "profit": 2.7,
    "profits": 2.8,
    "growth": 2.4,
    "loss": -2.7,
    "losses": -2.9,
    "decline": -2.4,
    "risk": -1.8,
    "uncertainty": -1.7,
    "bankruptcy": -3.5,
    "fraud": -4.0,
    "regulation": -1.6,
    "ban": -3.0,
    "lawsuit": -3.1,
}


def vader_pipeline(chunk):
  # creating a object
  sia = SentimentIntensityAnalyzer()

  # Merge custom lexicons
  CUSTOM_LEXICON = {}
  CUSTOM_LEXICON.update(CRYPTO_LEXICON)
  CUSTOM_LEXICON.update(FINANCIAL_LEXICON)
  # Inject into VADER
  sia.lexicon.update(CUSTOM_LEXICON)

  # run the polarity_score on the entire dataset and creating a score df containing the sentimental scores
  scores = chunk['processed_text'].progress_apply(sia.polarity_scores)
  scores_df = pd.DataFrame(list(scores))

  # merging each tweet with its sentiment score
  scores_df = scores_df.reset_index(drop=True)
  chunk = chunk.reset_index(drop=True)
  chunk = pd.concat([chunk, scores_df], axis=1)

  return chunk


def roberta_pipeline(chunk, batch_size=32, device=None):
    # Auto-detect device if not provided
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval().to(device)
    texts = chunk['processed_text'].tolist()

    probs_all = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="RoBERTa inference",
            leave=False
        ):
            enc = tokenizer(
                texts[i:i+batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            logits = model(**enc).logits
            probs_all.append(softmax(logits.cpu().numpy(), axis=1))

    scores = np.vstack(probs_all)

    chunk[['roberta_neg','roberta_neu','roberta_pos']] = scores

    # class index
    labels = np.argmax(scores, axis=1)

    # max probability = intensity
    intensity = np.max(scores, axis=1)

    # signed sentiment score (paper-style)
    roberta_score = np.select(
        [labels == 2, labels == 0],
        [intensity, -intensity],
        default=0.0   # neutral
    )

    chunk['roberta_score'] = roberta_score
    return chunk


def analyze_predict(chunk):
    if chunk.empty:
        return chunk
        
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] NLP: Formatting texts...")

    # date preprocessing
    chunk = date_preprocess(chunk)

    # text preprocessing (language-aware)
    chunk = preprocess_multilang_tweets(chunk)

    # dropping null value rows
    chunk = chunk.dropna(subset=['processed_text'])
    chunk = chunk[chunk['processed_text'].str.strip().str.len() > 0]
    
    if chunk.empty:
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] NLP: No valid processed texts remaining.")
        return chunk
        
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] NLP: Sample processed text: '{chunk['processed_text'].iloc[0][:50]}...'")
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] NLP: VADER Analysis on {len(chunk)} posts...")
    # vader sentiment analysis
    chunk = vader_pipeline(chunk)

    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] NLP: RoBERTa Analysis on {len(chunk)} posts...")
    # roberta sentiment analysis
    chunk = roberta_pipeline(chunk)

    # Keep ONLY required columns
    final_chunk = chunk[[
        'date',
        'text',
        'compound',          # VADER compound
        'roberta_score'      # RoBERTa score
    ]].copy()

    # Optional: sort by time inside chunk
    final_chunk = final_chunk.sort_values('date')

    return final_chunk