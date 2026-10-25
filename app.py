import html
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Smart Skincare Finder", page_icon="✨", layout="centered")

CSV_PATH = Path(__file__).parent / "products.csv"
REQUIRED_COLUMNS = {"name", "category", "price", "description"}
TOP_K = 5

SKIN_TYPES = ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
CONCERNS = ["Redness", "Acne", "Dryness", "Dullness", "Large pores", "Sensitivity",
            "Dark spots", "Fine lines", "Sun protection"]
TAG_KEYWORDS = ["oily", "dry", "combination", "normal", "sensitive", "acne", "redness",
                "dark spots", "fine lines", "pores", "fragrance free"]
EXAMPLES = ["Oily skin with breakouts", "Dry, sensitive and red", "Dull skin and dark spots"]

# (background, text) colours for each category pill
CATEGORY_COLORS = {
    "Cleanser": ("#E3F6F1", "#1E8E78"), "Moisturizer": ("#FFF1DB", "#B96F00"),
    "Serum": ("#F0E6FF", "#7A3FD1"), "Treatment": ("#FFE5E5", "#C63D3D"),
    "Sunscreen": ("#FFF8CC", "#9A7B00"), "Toner": ("#E3F1FF", "#2B6CB0"),
    "Exfoliant": ("#FFE9F3", "#C2307A"), "Eye Cream": ("#EDEBFF", "#4B44C4"),
    "Mask": ("#E7F7FF", "#1F7A9E"), "Lip Care": ("#FFE4EA", "#C23B5E"),
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    .stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .hero, .card {font-family: 'Poppins', sans-serif;}
    .stApp {background: linear-gradient(180deg, #FFF3F8 0%, #FFFFFF 380px);}
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 1.5rem; max-width: 780px;}

    .hero {
        background: linear-gradient(120deg, #FF9EBB 0%, #C9A7FF 100%);
        border-radius: 24px; padding: 34px 32px 28px; margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(201, 120, 190, 0.28);
    }
    .hero h1 {margin: 0; padding: 0; font-size: 2.1rem; font-weight: 700; color: #fff;}
    .hero p {margin: 8px 0 0; color: rgba(255,255,255,.95); font-size: 1rem;}
    .chip {
        display: inline-block; background: rgba(255,255,255,.25); color: #fff;
        border: 1px solid rgba(255,255,255,.55); padding: 3px 13px; border-radius: 999px;
        font-size: .78rem; font-weight: 500; margin-right: 6px;
    }

    .stTabs [data-baseweb="tab-list"] {gap: 8px; border-bottom: none;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px; padding: 6px 18px; background: #fff; border: 1px solid #F4E1EA; height: auto;
    }
    .stTabs [aria-selected="true"] {background: #FDE4EC; color: #D6457A; border-color: #F2B8CE;}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

    div[data-baseweb="select"] > div, .stTextInput input {border-radius: 12px !important;}
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #F27BA3, #B58BFF); color: #fff; border: none;
        border-radius: 12px; padding: .6rem 1rem; font-weight: 600;
    }
    [data-testid="stFormSubmitButton"] button:hover {filter: brightness(1.06); color: #fff;}
    .stButton button {border-radius: 999px; border: 1px solid #F2B8CE; font-size: .82rem;}

    .section-title {font-weight: 600; font-size: 1.1rem; color: #2A2A3C; margin: 18px 0 10px;}

    .card {
        background: #fff; border: 1px solid #F4E1EA; border-radius: 18px;
        padding: 18px 20px; margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(214, 69, 122, 0.07); transition: transform .15s, box-shadow .15s;
    }
    .card:hover {transform: translateY(-2px); box-shadow: 0 10px 24px rgba(214, 69, 122, 0.16);}
    .card.best {border: 2px solid #E8789F; background: linear-gradient(180deg, #FFF7FA, #FFFFFF);}
    .row {display: flex; gap: 14px; align-items: flex-start;}
    .rank {
        flex: 0 0 34px; height: 34px; border-radius: 50%; background: #FDE4EC; color: #D6457A;
        font-weight: 700; display: flex; align-items: center; justify-content: center;
    }
    .card.best .rank {background: #D6457A; color: #fff;}
    .main {flex: 1; min-width: 0;}
    .top {display: flex; justify-content: space-between; gap: 10px; align-items: flex-start;}
    .name {font-weight: 600; font-size: 1.02rem; color: #2A2A3C; line-height: 1.35;}
    .price {
        font-weight: 600; color: #D6457A; background: #FFF0F5; padding: 2px 12px;
        border-radius: 999px; white-space: nowrap;
    }
    .pills {margin: 7px 0 0;}
    .pill {
        display: inline-block; font-size: .68rem; font-weight: 600; letter-spacing: .05em;
        text-transform: uppercase; padding: 2px 10px; border-radius: 999px; margin-right: 6px;
    }
    .kb {background: #E8F0FF; color: #2F5BD6;}
    .best-pill {background: #D6457A; color: #fff;}
    .desc {color: #5A5A6E; font-size: .88rem; line-height: 1.55; margin: 10px 0;}
    .tag {
        display: inline-block; background: #FFF0F5; color: #B8346A; font-size: .74rem;
        padding: 2px 10px; border-radius: 999px; margin: 0 6px 4px 0;
    }
    .meter {display: flex; align-items: center; gap: 12px; margin-top: 8px;}
    .bar {flex: 1; background: #F5EAF0; border-radius: 999px; height: 8px; overflow: hidden;}
    .bar > div {background: linear-gradient(90deg, #F5A3BF, #D6457A); height: 100%; border-radius: 999px;}
    .pct {font-weight: 600; font-size: .8rem; color: #D6457A; min-width: 66px; text-align: right;}
    .note {font-size: .78rem; color: #9A9AAA; margin-top: 30px; text-align: center;}

    /* Keep a light look even when Streamlit is in dark mode */
    .stApp {color: #2A2A3C;}
    .stApp label p, [data-testid="stWidgetLabel"] p {color: #2A2A3C !important;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: #6A6A7E !important;}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-baseweb="select"] > div > div,
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important; border: 1px solid #F0D5E0 !important; color: #2A2A3C !important;
    }
    div[data-baseweb="select"] div, div[data-baseweb="select"] span {color: #2A2A3C !important;}
    div[data-baseweb="select"] svg {fill: #2A2A3C !important;}
    [data-baseweb="base-input"] {background: #FFFFFF !important;}
    div[data-baseweb="input"] {background: #FFFFFF !important; border-radius: 12px; border: 1px solid #F0D5E0;}
    .stTextInput input {background: #FFFFFF !important; color: #2A2A3C !important;}
    div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"] {
        background: #FFFFFF !important; color: #2A2A3C !important;
    }
    div[data-baseweb="popover"] li:hover {background: #FFF0F5 !important;}
    .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p {color: #5A5A6E !important;}
    .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p {color: #D6457A !important;}
    .stButton button {background: #FFFFFF; color: #2A2A3C;}
    .stButton button p {color: #2A2A3C !important;}
    [data-testid="stFormSubmitButton"] button p {color: #FFFFFF !important;}
    [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {color: #6A6A7E !important;}
    [data-testid="stForm"] {border: 1px solid #F4E1EA; border-radius: 18px; background: #FFFFFF;}
    .stats {margin: -4px 0 14px; display: flex; gap: 8px; flex-wrap: wrap;}
    .stat {
        background: #FFFFFF; border: 1px solid #F4E1EA; color: #5A5A6E; border-radius: 999px;
        padding: 4px 14px; font-size: .8rem;
    }
    .stat b {color: #D6457A;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load products from products.csv
# ---------------------------------------------------------------------------
@st.cache_data
def load_products(path: str, modified: float) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    data.columns = [c.strip().lower() for c in data.columns]
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"products.csv is missing column(s): {', '.join(sorted(missing))}")
    if "origin" not in data.columns:
        data["origin"] = "Other"
    data = data.dropna(subset=["name", "category", "price", "description"])
    data["origin"] = data["origin"].fillna("Other")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    return data.dropna(subset=["price"]).reset_index(drop=True)


if not CSV_PATH.exists():
    st.error("Could not find products.csv. Put it in the same folder as app.py.")
    st.stop()

try:
    df = load_products(str(CSV_PATH), CSV_PATH.stat().st_mtime)
except Exception as e:
    st.error(f"Problem reading products.csv: {e}")
    st.stop()

if df.empty:
    st.error("products.csv has no usable rows.")
    st.stop()

CATEGORIES = ["All"] + sorted(df["category"].unique())


# ---------------------------------------------------------------------------
# Model + embeddings
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Preparing products...")
def embed_descriptions(descriptions):
    return load_model().encode(list(descriptions), normalize_embeddings=True)


vectors = embed_descriptions(tuple(df["description"]))


def recommend(query, max_price, category="All", korean_only=False, top_k=TOP_K):
    q_vec = load_model().encode([query], normalize_embeddings=True)[0]
    out = df.copy()
    out["match"] = vectors @ q_vec
    out = out[out["price"] <= max_price]
    if category != "All":
        out = out[out["category"] == category]
    if korean_only:
        out = out[out["origin"] == "Korean"]
    return out.sort_values("match", ascending=False).head(top_k)


def show_results(results: pd.DataFrame):
    if results.empty:
        st.warning("No products found for those filters. Try a higher budget or another product type.")
        return
    st.markdown(f'<div class="section-title">Your top {len(results)} matches</div>', unsafe_allow_html=True)
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        pct = int(np.clip(row["match"], 0, 1) * 100)
        desc_lower = str(row["description"]).lower()
        tags = "".join(f'<span class="tag">{t.title()}</span>' for t in TAG_KEYWORDS if t in desc_lower)
        bg, fg = CATEGORY_COLORS.get(row["category"], ("#EEEEF2", "#555566"))
        pills = f'<span class="pill" style="background:{bg};color:{fg}">{html.escape(str(row["category"]))}</span>'
        if row["origin"] == "Korean":
            pills += '<span class="pill kb">🇰🇷 K-Beauty</span>'
        if rank == 1:
            pills += '<span class="pill best-pill">Best match</span>'
        name = html.escape(str(row["name"]))
        description = html.escape(str(row["description"]))
        price = f"{row['price']:g}"
        best_cls = " best" if rank == 1 else ""
        # One unbroken string: blank or indented lines would make Markdown show it as code
        card_html = (
            f'<div class="card{best_cls}"><div class="row">'
            f'<div class="rank">{rank}</div>'
            '<div class="main">'
            f'<div class="top"><div class="name">{name}</div><div class="price">&#36;{price}</div></div>'
            f'<div class="pills">{pills}</div>'
            f'<div class="desc">{description}</div>'
            f"<div>{tags}</div>"
            f'<div class="meter"><div class="bar"><div style="width:{pct}%"></div></div>'
            f'<div class="pct">{pct}% match</div></div>'
            "</div></div></div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero + shared filters
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>✨ Smart Skincare Finder</h1>
        <p>Describe your skin and get product matches ranked by meaning, not just keywords.</p>
    </div>
    """.replace("\n", ""),
    unsafe_allow_html=True,
)

if "visits" not in st.session_state:
    st.session_state.visits = 1
    st.session_state.start = time.time()


@st.fragment(run_every=1)
def session_stats():
    secs = int(time.time() - st.session_state.start)
    st.markdown(
        f'<div class="stats"><span class="stat">Visit count: <b>{st.session_state.visits}</b></span>'
        f'<span class="stat">Time on page: <b>{secs} s</b></span></div>',
        unsafe_allow_html=True,
    )


session_stats()

with st.container(border=True):
    f1, f2 = st.columns([3, 2])
    category = f1.selectbox("Product type", CATEGORIES)
    f2.write("")
    korean_only = f2.toggle("🇰🇷 K-Beauty only", value=False)

tab_guided, tab_search = st.tabs(["✨ Personalized match", "🔎 Describe your skin"])

with tab_guided:
    with st.form("guided"):
        c1, c2 = st.columns(2)
        skin_type = c1.selectbox("Skin type", SKIN_TYPES)
        concern = c2.selectbox("Main concern", CONCERNS)
        max_budget = int(max(df["price"].max(), 50))
        budget = st.slider("Maximum budget ($)", 5, max_budget, 25)
        submitted = st.form_submit_button("Get Recommendations", use_container_width=True)
    if submitted:
        noun = "skincare product" if category == "All" else category.lower()
        query = f"{noun} for {skin_type.lower()} skin with {concern.lower()}"
        show_results(recommend(query, budget, category, korean_only))


def set_example(text):
    st.session_state.free_text = text


with tab_search:
    st.caption("Try an example:")
    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        col.button(ex, on_click=set_example, args=(ex,), use_container_width=True)
    free_text = st.text_input(
        "Tell me your skin type or skin concern",
        key="free_text",
        placeholder="e.g. oily skin that gets red and irritated easily",
    )
    if free_text:
        show_results(recommend(free_text, float(df["price"].max()), category, korean_only))

st.markdown(
    '<div class="note">Recommendations are for general guidance only. '
    "See a dermatologist for persistent skin problems.</div>",
    unsafe_allow_html=True,
)
