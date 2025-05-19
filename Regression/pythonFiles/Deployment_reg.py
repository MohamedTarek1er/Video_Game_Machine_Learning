import streamlit as st
import numpy as np
import pandas as pd
import joblib
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from catboost import CatBoostRegressor

nltk.download('stopwords')
nltk.download('wordnet')

def load_assets():
    model = CatBoostRegressor()
    model.load_model(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\model1C.cbm')

    robust_scaler = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\robust_scaler.pkl')
    minmax_scaler = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\minmax_scaler.pkl')
    onehot_encoder = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\onehot_encoder.pkl')
    mlb_genres = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\mlb_genres_encoder.pkl')
    mlb_platforms = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\mlb_platforms_encoder.pkl')
    dict1_genres = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\dict1_genres.pkl')
    dict2_platforms = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\dict2_platforms.pkl')
    dict12_names = joblib.load(r'C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\Regression\Models\dict12_names.pkl')

    return model, robust_scaler, minmax_scaler, onehot_encoder, mlb_genres, mlb_platforms, dict1_genres, dict2_platforms, dict12_names

model, robust_scaler, minmax_scaler, onehot_encoder, mlb_genres, mlb_platforms, dict1_genres, dict2_platforms, dict12_names = load_assets()

def clean_and_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_words = []
    for word in words:
        word = word.strip().lower()
        word = re.sub(r'[^a-zA-Z]', '', word)
        if word and word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)
    return cleaned_words

st.title("🎮 Predict Video Game Copies Sold")

name = st.text_input("Enter Game Name")

steam_achievements = st.checkbox("Steam Achievements")
steam_trading_cards = st.checkbox("Steam Trading Cards")
workshop_support = st.checkbox("Workshop Support")

genre_options = mlb_genres.classes_.tolist()
genres = st.multiselect("Select Genres", options=genre_options)

release_date = st.date_input("Release Date")
age_years = 2026 - release_date.year
age_years = max(age_years, 0)

platform_options = mlb_platforms.classes_.tolist()
supported_platforms = st.multiselect("Supported Platforms", options=platform_options)

price = st.number_input("Enter Price ($):", min_value=0.0, step=1.00)

publisher_options = ["Hobbyist", "AA", "AAA"]
publisherClass = st.selectbox("Select Publisher Class:", publisher_options)

reviewScore = st.slider("Enter Review Score (0-100):", min_value=0, max_value=100)

# ----------------- Feature Engineering -----------------
extras_mean = ((int(steam_achievements) + int(steam_trading_cards) + int(workshop_support)) / 3) + 1

pdict1 = { "Hobbyist": 0.01, "Indie": 0.5, "AA": 3, "AAA": 10 }
publisher_value = pdict1.get(publisherClass, 0.01)

game_rating = extras_mean * (reviewScore + 1) * (age_years + 1) * publisher_value

if genres:
    genre_mean = np.mean([dict1_genres.get(genre, 0) for genre in genres])
else:
    genre_mean = 0
genre_mean += 1
game_rating_with_genres = (genre_mean * game_rating) / 1e7

rating_over_price = game_rating_with_genres / (price + 1)

platform_sum = sum([dict2_platforms.get(p, 0) for p in supported_platforms])
game_rating_with_platforms = rating_over_price * platform_sum

name_cleaned = clean_and_lemmatize(name)
if name_cleaned:
    name_mean = np.mean([dict12_names.get(word, 0) for word in name_cleaned])
else:
    name_mean = 0
name_mean += 1
name_as_copies_sold = name_mean / 1e4

game_rating_with_names = name_as_copies_sold * game_rating_with_platforms

# ----------------- Encoding and Scaling -----------------
genre_selected = ["Action", "Casual", "Free To Play", "Indie", "Massively Multiplayer"]
genre_encoding = [1 if g in genres else 0 for g in genre_selected]

linux_encoded = 1 if "linux" in supported_platforms else 0

steam_achievements_encoded = [int(steam_achievements), int(not steam_achievements)]
steam_trading_cards_encoded = [int(steam_trading_cards), int(not steam_trading_cards)]
workshop_support_encoded = [int(workshop_support), int(not workshop_support)]

publisher_classes = ["Hobbyist", "AA", "AAA"]
publisher_encoding = [1 if publisherClass == p else 0 for p in publisher_classes]
    
raw_features = np.array([
    price,
    age_years,
    game_rating,
    game_rating_with_genres,
    rating_over_price,
    game_rating_with_platforms,
    name_as_copies_sold,
    game_rating_with_names,
    *genre_encoding,
    linux_encoded,
    *steam_achievements_encoded,
    *steam_trading_cards_encoded,
    *workshop_support_encoded,
    *publisher_encoding
]).reshape(1, -1)

feature_names = [
    "price","age_years", "1-GameRating", "2-GameRatingWithGenres", "3-RatingOverPrice",
    "4-GameRatingWithPlatforms", "5-NameAsCopiesSold", "6-GameRatingWithNames",
    "Action", "Casual", "Free To Play", "Indie", "Massively Multiplayer",
    "linux",
    "steam_achievements_1", "steam_achievements_2",
    "steam_trading_cards_1", "steam_trading_cards_2",
    "workshop_support_1", "workshop_support_2",
    "publisherClass_1", "publisherClass_3", "publisherClass_4"
]

df_features = pd.DataFrame(raw_features, columns=feature_names)

numerical_columns=["price","1-GameRating","2-GameRatingWithGenres","4-GameRatingWithPlatforms",
                   "3-RatingOverPrice","5-NameAsCopiesSold","6-GameRatingWithNames"]

df_features[numerical_columns] = robust_scaler.transform(df_features[numerical_columns])
review_score_scaled = minmax_scaler.transform([[reviewScore]])[0][0]
df_features["reviewScore"] = review_score_scaled

df_features.drop(columns=["price"], inplace=True)

if st.button("Predict Copies Sold"):
    prediction = model.predict(df_features)[0]
    st.success(f"🎯 Predicted Copies Sold: {prediction:,.2f}")
