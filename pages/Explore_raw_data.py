import streamlit as st
import pandas as pd

# ------------------ קבועים ------------------
NY_URL = "https://raw.githubusercontent.com/sam-israel/general/refs/heads/master/listings%20NYC.csv"
LA_URL = "https://raw.githubusercontent.com/sam-israel/general/refs/heads/master/listings%20LA.csv"

st.write("Polaris Group")
st.title("Airbnb – NYC + LA (Raw Data)")

# ------------------ טעינת דאטה ------------------
@st.cache_data
def load_data():
    df_ny = pd.read_csv(NY_URL)
    df_la = pd.read_csv(LA_URL)

    df_ny.insert(0, "city", "NY")
    df_la.insert(0, "city", "LA")

    df = pd.concat([df_ny, df_la], ignore_index=True)

    # טיפוסים לפילטרים
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
    df["review_scores_rating"] = pd.to_numeric(df["review_scores_rating"], errors="coerce")

    return df

df_all = load_data()

# ------------------ עמודות להצגה ------------------
cols_to_show = [
    "review_scores_rating",
    "city",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "price",
    "name",
    "description"
]

# ------------------ ברירות מחדל ------------------
rating_min_val = int(df_all["review_scores_rating"].dropna().min())
rating_max_val = int(df_all["review_scores_rating"].dropna().max())

DEFAULTS = {
    "city": "הכול",
    "bathrooms": "הכול",
    "bedrooms": "הכול",
    "rating_range": (rating_min_val, rating_max_val),
    "include_rating_nan": True,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------ Sidebar ------------------
st.sidebar.header("פילטרים")

# כפתור איפוס
if st.sidebar.button("איפוס פילטרים"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# ----- עיר -----
city_options = ["הכול", "ללא ערך (NaN)"] + sorted(df_all["city"].dropna().unique())
st.sidebar.selectbox("עיר", city_options, key="city")

# ----- חדרי רחצה -----
bathroom_options = ["הכול", "ללא ערך (NaN)"] + sorted(df_all["bathrooms"].dropna().unique())
st.sidebar.selectbox("מספר חדרי רחצה", bathroom_options, key="bathrooms")

# ----- חדרים -----
bedroom_options = ["הכול", "ללא ערך (NaN)"] + sorted(df_all["bedrooms"].dropna().unique().astype(int))
st.sidebar.selectbox("מספר חדרים", bedroom_options, key="bedrooms")

# ----- דירוגים -----
st.sidebar.checkbox("כלול גם דירוג חסר (NaN)", key="include_rating_nan")

st.sidebar.slider(
    "טווח דירוגים",
    min_value=rating_min_val,
    max_value=rating_max_val,
    key="rating_range"
)

# ------------------ סינון בפועל ------------------
df_filtered = df_all.copy()

# עיר
if st.session_state.city == "ללא ערך (NaN)":
    df_filtered = df_filtered[df_filtered["city"].isna()]
elif st.session_state.city != "הכול":
    df_filtered = df_filtered[df_filtered["city"] == st.session_state.city]

# חדרי רחצה
if st.session_state.bathrooms == "ללא ערך (NaN)":
    df_filtered = df_filtered[df_filtered["bathrooms"].isna()]
elif st.session_state.bathrooms != "הכול":
    df_filtered = df_filtered[
        df_filtered["bathrooms"].notna() &
        (df_filtered["bathrooms"] == st.session_state.bathrooms)
    ]

# חדרים
if st.session_state.bedrooms == "ללא ערך (NaN)":
    df_filtered = df_filtered[df_filtered["bedrooms"].isna()]
elif st.session_state.bedrooms != "הכול":
    df_filtered = df_filtered[
        df_filtered["bedrooms"].notna() &
        (df_filtered["bedrooms"] == st.session_state.bedrooms)
    ]

# דירוגים
r_min, r_max = st.session_state.rating_range
if st.session_state.include_rating_nan:
    df_filtered = df_filtered[
        df_filtered["review_scores_rating"].isna() |
        df_filtered["review_scores_rating"].between(r_min, r_max)
    ]
else:
    df_filtered = df_filtered[
        df_filtered["review_scores_rating"].notna() &
        df_filtered["review_scores_rating"].between(r_min, r_max)
    ]

# ------------------ חיווי + טבלה ------------------
st.sidebar.metric("נכסים אחרי סינון", len(df_filtered))
st.sidebar.caption(f"הוסרו {len(df_all) - len(df_filtered)} נכסים")

st.data_editor(
    df_filtered[cols_to_show],
    use_container_width=True,
    num_rows="dynamic"
)

