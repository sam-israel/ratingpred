import os
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from openai import OpenAI, RateLimitError

DEFAULT_DATASET1_LOC = 'https://raw.githubusercontent.com/sam-israel/general/refs/heads/master/listings%20NYC.csv'
DEFAULT_DATASET2_LOC = 'https://raw.githubusercontent.com/sam-israel/general/refs/heads/master/listings%20LA.csv'
DEFAULT_DATASET3_LOC = 'https://github.com/sam-israel/general/raw/refs/heads/master/TEST_SET_newcity1.csv'
DEFAULT_OUTPUT_LOC = "data"
WANDB_API_KEY = os.getenv('WANDB_API_KEY', default="")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", default="")
MODEL="openai/gpt-oss-20b"   #meta-llama/Meta-Llama-3.1-8B-Instruct-fast"
BATCH_SIZE = 40
MAX_WORKERS = 4
SYSTEM_PROMPT = """
    You are a strict travel-review scoring assistant.

    Score the provided TEXT on these dimensions (integer 1-10):
    1) center:
       How central/convenient the location is.
       1 = very far/inconvenient, 10 = very central.
    2) quiet:
       How quiet the place is.
       1 = very noisy, 10 = very quiet.
    3) facilities:
       Quality/completeness of apartment facilities (e.g., kitchen, AC/heating, washer, Wi-Fi, parking, elevator, cleanliness-related setup).
       1 = very poor/minimal facilities, 10 = excellent/well-equipped facilities.

    Rules:
    - Use only information implied by the text.
    - If information is unclear, give a conservative mid score (5 or 6), not null.
    - Return JSON only in this schema:
        {"results":[{"id":<int>,"center":<int 1-10>,"quiet":<int 1-10>,"facilities":<int 1-10>}]}
      No extra keys, no prose.
""".strip()
CENTRAL_PROTOTYPES = [
    "in the city center",
    "in the heart of downtown",
    "easy access to main attractions",
    "walking distance to major city landmarks",
    "prime location"
]



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}



def preprocess(df:DataFrame, y_col, drop_duplicate_rows=True, use_genAI=False, verbose=True):
    if verbose:
      print(f"Initial rows number {df.shape[0]}")

    # Correct data types
    clean_airbnb_schema(df, inplace=True)

    if verbose:
      print(f"After correcting data types: {df.shape[0]}")
    
    # Drop duplicate rows
    if drop_duplicate_rows:
      df= drop_dup_rows(df)
      if verbose:
        print(f"After cleaning duplicate rows : {df.shape[0]}")

    # Drop rows with null y values
    if y_col is not None:
      df = drop_y_null(df, y_col )

    if verbose:
      print(f"After dropping rows with null y values : {df.shape[0]}")

    # Drop redundant columns
    df = drop_redunt_cols(df)

    # use genAI to analyze text fields
    if use_genAI:
      print("Analyzing with ai...")  
      df = analyze_text(df)
    else:
      df = calc_central_score(df)
    
    # Feature engineering
    print("Feature transformation")
    df = transformation (df, y_col )

    return df

def clean_airbnb_schema(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    "Convert into correct data types"

    if not inplace:
        df = df.copy()

    # Dates -> datetime64[ns]
    date_cols = ["last_scraped", "host_since", "first_review", "last_review"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Percentages -> float64 (range 0–1)
    pct_cols = ["host_response_rate", "host_acceptance_rate"]
    for col in pct_cols:
        if col in df.columns:
            s = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace("%", "", regex=False)
            )
            num = pd.to_numeric(s, errors="coerce").astype("Float64")
            df[col] = num.where(num <= 1, num / 100).astype("float64")

    # Booleans -> pandas nullable boolean
    bool_cols = ["host_is_superhost", "host_has_profile_pic", "instant_bookable"]

    true_set = {"t", "true", "1", "yes", "y"}
    false_set = {"f", "false", "0", "no", "n"}

    def to_bool(x):
        if pd.isna(x):
            return pd.NA
        v = str(x).strip().lower()
        if v in true_set:
            return True
        if v in false_set:
            return False
        return pd.NA

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(to_bool).astype("boolean")

    # Price -> float64
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype("string")
            .str.strip()
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")

    # Counts with missing values -> Int64 (nullable integer)
    nullable_int_cols = ["host_listings_count", "host_total_listings_count", "bedrooms"]
    for col in nullable_int_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if (s.dropna() % 1 != 0).any():
                df[col] = s.astype("float64")
            else:
                df[col] = s.astype("Int64")

    # Text columns -> pandas string
    text_cols = [
        "name",
        "description",
        "neighborhood_overview",
        "host_name",
        "host_about",
        "bathrooms_text",
        "amenities",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if verbose:
        print(df.dtypes)

    return df


def concat_datasets(df1, df2):
    "Concat two datasets"

    return pd.concat([df1, df2], ignore_index=True, sort=False, axis=0)


def drop_dup_rows(df):
    """Drop duplicate Rows"""
    return df.drop_duplicates()


def drop_redunt_cols(df):
    """Drop  redundant columns"""
    df_copy = df.copy()
    cols_to_drop =  [ "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights","minimum_nights_avg_ntm",  "maximum_nights_avg_ntm" ]
      
    df_copy.drop(cols_to_drop, axis=1, inplace=True)

    return df_copy

def transformation(arr, y_col):
  """Feature engeneering"""

  if y_col is None:
    X = arr.copy()
  else:
    y = arr.loc[:,y_col]
    X = arr.copy().drop(columns=y_col)

  for i in X.select_dtypes(include=["string", "object"]).columns:  # Convert each text into its length
    X[i] = X[i].apply(lambda x: 0 if pd.isna(x) else len(x))

  for i in X.select_dtypes(include=["datetime"]): # Convert each date into the number of days since 1/1/2000
    t0 = pd.Timestamp("2000-01-01")
    X[i] = (X[i] - t0).dt.days.astype("Int32")

  for i in X.select_dtypes(include=["bool"]).columns: # Convert bool to Int (technical)
      X[i] = X[i].astype("Int32")

  if y_col is None:
    return X
  else:  
    return pd.concat([X,y], axis=1)


def drop_y_null(df, y_col ):
    """Drop rows with null y values"""
    y = df.loc[:,y_col]
    
    mask = y.notna()
    
    return df.loc[mask,:]


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def calc_central_score(df):
    """
    calc central location score based on cousine similarity between the apt description and some "central" phrases
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text_list = df["description"].fillna("").astype(str).tolist()
    all_emb = []
    chunk_size = 500
    for i in range(0, len(text_list), chunk_size):
        to = i + chunk_size
        if (to > len(text_list)):
            to = len(text_list)
        chunk = text_list[i:to]
        emb = model.encode(
            chunk,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_emb.append(emb)
#        print (to)

    emb_texts = np.vstack(all_emb)  # shape: (50000, dim)
    emb_central = model.encode(CENTRAL_PROTOTYPES, normalize_embeddings=True)

    # calc cosine similarity in [-1, 1] and grade 1-10
    S = cosine_similarity(emb_texts, emb_central)
    sim = S.max(axis=1)
    df["central"] = np.clip(1 + 9 * ((sim + 1) / 2), 1, 10).round(2)
#    print (print(df.loc[:20, ["description", "central"]]))
    return df


def score_batch(items):
    """
    items: list of dicts [{"id": 12, "text": "..."}]
    returns dict id -> scores
    """
    user_payload = {"items": items}

    client = OpenAI(api_key=NEBIUS_API_KEY, base_url = "https://api.tokenfactory.nebius.com/v1")


    for _ in range(3):
        try:
            # Chat Completions style for gpt-3.5-turbo:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)

            out = {}
            for r in data.get("results", []):
                rid = int(r["id"])
                out[rid] = {
                    "center": max(1, min(10, int(r["center"]))),
                    "quiet": max(1, min(10, int(r["quiet"]))),
                    "facilities": max(1, min(10, int(r["facilities"]))),
                }
            return out

        except RateLimitError:
            time.sleep(1.5) # seconds
    return {}

def analyze_text(df) :
    """ Analyze apartment descriptions and convert them into 3 numeric scores, using openAI """
    df["text"] = df["description"].fillna("") + df["amenities"].fillna("") + df["neighborhood_overview"].fillna("")

    # prepare rows
    rows = [{"id": int(i), "text": (t if isinstance(t, str) else "")}
            for i, t in zip(df.index, df["text"])]

    # optional dedup cache by exact text to save calls
    text_to_score = {}
    id_to_text = {r["id"]: r["text"] for r in rows}

    # score unique texts in parallel batches
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for b in chunked(rows, BATCH_SIZE):
            futures.append(ex.submit(score_batch, b))

        for f in as_completed(futures):
            res = f.result()
            for id, score in res.items():
                text_to_score[id_to_text[id]] = score

    # map back to dataframe rows
    centers, quiets, facs = [], [], []
    for idx in df.index:
        txt = id_to_text[int(idx)]
        s = text_to_score.get(txt, None)
        if s is None:
            centers.append(pd.NA); quiets.append(pd.NA); facs.append(pd.NA)
        else:
            centers.append(s["center"]); quiets.append(s["quiet"]); facs.append(s["facilities"])

    df["center"] = pd.Series(centers, index=df.index, dtype="Int64")
    df["quiet"] = pd.Series(quiets, index=df.index, dtype="Int64")
    df["facilities"] = pd.Series(facs, index=df.index, dtype="Int64")
    df.drop(columns=["text"])
    return df
