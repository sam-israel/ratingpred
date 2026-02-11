import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split



DEFAULT_DATASET1_LOC = 'https://raw.githubusercontent.com/sam-israel/general/refs/heads/master/listings%20NYC.csv'
DEFAULT_DATASET2_LOC = 'https://raw.githubusercontent.com/sam-israel/general/refs/heads/master/listings%20LA.csv'
DEFAULT_OUTPUT_LOC = "data"
Y_COL = "review_scores_rating"


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


def concat_datasets(df1, df2, df1_citycode=1, df2_citycode=0):
    "Concat two datasets"
    df1.insert(0, "city", df1_citycode)
    df2.insert(0, "city", df2_citycode)

    return pd.concat([df1, df2], ignore_index=True, sort=False)




def drop_dup_rows(df):
    """Drop duplicate Rows"""
    return df.drop_duplicates()


def drop_redunt_cols(df):
    """Drop  redundant columns"""
    return df.drop(  columns= [ "minimum_minimum_nights",
    "maximum_minimum_nights",
     "minimum_maximum_nights",
    "maximum_maximum_nights","minimum_nights_avg_ntm",  "maximum_nights_avg_ntm" ])


def transformation(arr, y_col = Y_COL):
  """Feature engeneering"""

  if y_col is None:
    X = arr.copy()
  else:
    y = arr.loc[:,y_col]
    X = arr.copy().drop(columns=y_col)

  for i in X.select_dtypes("number").columns:
    X[i + "_isNA"] = X[i].isna() # Make an indicator column for NA values in numeric columns
    q_high = arr[i].quantile(0.995)
    X[i + "_isOutlier"] = (~X[i].isna()) &  (X[i] > q_high) # Make an indicator column high numeric values

  for i in X.select_dtypes(include=["string", "object"]).columns:  # Convert each text into its length
    X[i] = X[i].apply(lambda x: 0 if pd.isna(x) else len(x))

  for i in X.select_dtypes(include=["datetime"]): # Convert each date into the number of days since 1/1/2000
    t0 = pd.Timestamp("2000-01-01")
    X[i] = (X[i] - t0).dt.days

  for i in X.select_dtypes(include=["bool"]).columns: # Convert bool to Int (technical)
      X[i] = X[i].astype("Int32")

  if y_col is None:
    return X
  else:  
    return pd.concat([X,y], axis=1)



def drop_y_null(df, y_col = Y_COL):
    """Drop rows with null y values"""
    y = df.loc[:,Y_COL]
    
    mask = y.notna()
    
    return df.loc[mask,:]



