import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
from category_encoders import OneHotEncoder,BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split
import re
import ast
from scipy.stats import pointbiserialr
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor
import missingno as msno
from sklearn.preprocessing import RobustScaler
import matplotlib.patches as mpatches
import math
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

def data_info(data):

    """
    This function returns a DataFrame containing the summary information for each column 
    """

    Names=[col for col in data]
    data_types=[data[col].dtype for col in data.columns]
    top_10_unique_values=[data[col].value_counts().head(10).index.to_list() for col in data.columns]
    nunique_values=[data[col].nunique() for col in data.columns]
    nulls=[data[col].isnull().sum() for col in data.columns]
    percent_of_Nulls= [data[col].isnull().sum()/len(data)*100 for col in data.columns]
    duplicates=data.duplicated().sum()


    info_df=pd.DataFrame({'Name':Names,
                          'Data_Type':data_types,
                          'Top_10_Unique_Values':top_10_unique_values,
                          'Nunique_Values':nunique_values,
                          'Nulls':nulls,
                          'Percent_of_Nulls':percent_of_Nulls,
                          'Duplicates':duplicates})
    return info_df


# # Reading Data

df1=pd.read_csv(r"C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\info_base_games.csv")
print(df1.head())
print(df1.shape)
print(data_info(df1))

df2=pd.read_csv(r"C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\demos.csv")
print(df2.head())
print(df2.shape)
print(data_info(df2))

df3=pd.read_csv(r"C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\dlcs.csv")
print(df3.head())
print(df3.shape)
print(data_info(df3))

df4=pd.read_csv(r"C:\Users\moham\Downloads\College\Machine_Learning\PROJECT\gamalytic_steam_games.csv")
print(df4.head())
print(df4.shape)
print(data_info(df4))

# # Convert Appid Data Type in all Data Frames

df1["appid"] = pd.to_numeric(df1['appid'], errors='coerce').astype('Int64')
df2["appid"] = pd.to_numeric(df2['full_game_appid'], errors='coerce').astype('Int64')
df3["appid"] = pd.to_numeric(df3['base_appid'], errors='coerce').astype('Int64')
df4["appid"] = pd.to_numeric(df4['steamId'], errors='coerce').astype('Int64')

# # Merge All Data Frames

merged = pd.merge(df1,   df4, on='appid', how='inner')
merged = pd.merge(merged, df2,   on='appid', how='left')
df = pd.merge(merged, df3,  on='appid', how='left')

print(data_info(df))

# # Dropping Columns With High Null%

dropped_list=["metacritic","achievements_total","aiContent","Unnamed: 0","full_game_appid","demo_appid","name_y",
              "base_appid","dlc_appid","name"]

df.drop(columns=dropped_list,inplace=True)

print(data_info(df))

# # Measure Correlation with id's

list1=["appid","steamId","copiesSold"]
cor=df[list1].corr()
print(cor)

# # Dropping columns with high unique values and irrelevant to target

df.drop(["steamId","appid"],axis=1,inplace=True)
print(data_info(df))

# # Preprocess on all merged Data (Handle Nulls, Duplicates , Outliers)

# Missing Values are in Genres & release_date column

df["genres"]=df["genres"].fillna(method='ffill').fillna(method='bfill')
df.dropna(subset=["release_date"], inplace=True)

print(data_info(df))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# # Converting release_date to age by years (>=2026 → 0, 2025 → 1, 2024 → 2, etc.)

year_anchor = pd.Timestamp('2026-01-01')

df['age_years'] = df["release_date"].str.replace(r".*-25$", "Jan 1, 2025", regex=True)
df['age_years'] = df['age_years'].apply(lambda x: "Jan 1, 2025" if x in ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025", "2025"] else x)
df['age_years'] = pd.to_datetime(df['age_years'], errors='coerce')
df['age_years'] = df['age_years'].fillna(year_anchor)
df['age_years'] = (year_anchor.year - df['age_years'].dt.year)
df.drop(columns=['release_date'], inplace=True)

print(data_info(df))

# # Handling Outliers

# def handle_Numerical_outliers(data,column):
#     """
#     This function handles outliers.
#     """
#     i = 1
#     plt.figure(figsize=(15, 8))
#     for col in column:
#         plt.subplot(2, 2 , i)
#         sns.boxplot(y=data[col], color='skyblue')
#         plt.title(col)
#         i += 1
#     plt.tight_layout()
#     plt.suptitle('Boxplots Before Handling Outliers', fontsize=18, y=1.02)
#     plt.show()

#     i=1

#     for col in column:
#         Q1 = data[col].quantile(0.25)
#         Q3 = data[col].quantile(0.75)
#         IQR = Q3 - Q1

#         lower_bound = Q1 - 1.9 * IQR
#         upper_bound = Q3 + 1.5 * IQR

#         data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

#     plt.figure(figsize=(15, 8))
#     for col in column:
#         plt.subplot(2, 2 , i)
#         sns.boxplot(y=data[col], color='skyblue')
#         plt.title(col)
#         i += 1
#     plt.tight_layout()
#     plt.suptitle('Boxplots After Handling Outliers', fontsize=18, y=1.02)
#     plt.show()
    
#     return data

# li=["price","reviewScore"]

# df=handle_Numerical_outliers(df,li)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Feature Engineering

# ## 🎮 1. GameRating
# 
# We engineered the `GameRating` feature by combining multiple signals that correlate with game sales. Each sub-feature is designed to reflect real-world factors affecting game performance.
# 
# ### 🧩 Components:
# 
# #### ✅ `extras_mean`
# - Mean of game-related extras: **Achievements**, **Trading Cards**, and **Workshop Support**.
# - A small constant (+1) is added to avoid zero values.
# - 🧠 **Intuition**: More extras typically lead to higher player engagement → **More Sales** *(Direct Relation)*.
# 
# #### ✅ `reviewScore`
# - Represents the overall review score of the game.
# - Also incremented by +1 to prevent zero values.
# - 🧠 **Intuition**: Better reviews attract more players → **More Sales** *(Direct Relation)*.
# 
# #### ✅ `publisher_encode`
# - Numerical encoding of publisher type:
#   - AAA >>> AA >> Indie > Hobbyist
# - 🧠 **Intuition**: Well-known publishers usually have more marketing power → **More Sales**.
# 
# #### ✅ `age_years`
# - Release date converted to age by years (2026 and above = 0, 2025 = 1, 2024 = 2, etc.).
# - +1 to avoid multiplication by zero.
# - 🧠 **Intuition**: Older games have had more time to accumulate sales → **Inverse Relation**  
#   *(i.e., earlier release = more time for copies sold)*.

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

publisher_encode = df["publisherClass"].map({ "Hobbyist": 0.01, "Indie": 0.5, "AA": 3, "AAA": 10 })
extras_mean=(df["steam_achievements"]+df["steam_trading_cards"]+df["workshop_support"])/3

df["1-GameRating"]=((extras_mean+1)*(df["reviewScore"]+1)*publisher_encode)*(df['age_years'] + 1)

# # Measure Correlation with Target

list1=["1-GameRating","copiesSold"]
correlation=df[list1].corr()
print(correlation)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ## 📊 2. GameRatingWithGenres
# 
# - Included the `genres` column in our `GameRating` feature.  
# - Slightly worse correlation from **0.209** → **0.202**.
# 
# ### 🧩 Steps:
# 1. Get total `copiesSold` for each unique genre across the dataframe.
# 2. Replace every row in `genres` with the **mean** of `copiesSold` of its genres.
# 3. Divide by **10 million** to make the values smaller.
# 4. Multiply `GameRating` by the new `genres` value to create the new feature.

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

df['genresTemp'] = df['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')])
from collections import Counter

genre_counter = Counter([genre for sublist in df['genresTemp'] for genre in sublist])
common_genres = {genre for genre, count in genre_counter.items()}

dict1={}
for i in common_genres:
    dict1[i]=0
for idx,genre in df["genresTemp"].items():
    for j in genre:
        dict1[j] += df.loc[idx, "copiesSold"]

genre_means = []

for idx, genre_list in df["genresTemp"].items():
    mean1 = 0
    for genre in genre_list:
        mean1 += dict1[genre]
    mean1 /= len(genre_list)
    genre_means.append(mean1)

df.drop("genresTemp",inplace=True,axis=1)
genre_means = [x + 1 for x in genre_means]
df["2-GameRatingWithGenres"] = (genre_means * df["1-GameRating"])/1e7

# # Measure Correlation with Target

li=["2-GameRatingWithGenres","copiesSold"]
cor=df[li].corr()
print(cor)
# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ## 💰 3. RatingOverPrice
# 
# - Divided `GameRatingWithGenres` feature by `price` feature.
# - +1 to avoid division by zero.
# - Improves correlation from **0.202** → **0.389**.
# - 🧠 **Intuition**: Lower price (generally) means more sales → Inverse Relation

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

df["3-RatingOverPrice"]=df["2-GameRatingWithGenres"]/(df["price"]+1)

print(df["publisherClass"].value_counts())

# # Measure Correlation with Target

li=["3-RatingOverPrice","copiesSold"]
cor=df[li].corr()
print(cor)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ## 💻 4. GameRatingWithPlatforms
# 
# - Included the `supported_platforms` column in our `RatingOverPrice` feature.
# - Improves correlation from **0.389** → **0.584**.
# 
# ### 🧩 Steps:
# 1. Set each platform to a specific value (trial & errored our choices).
# 2. Replaced each value in `supported_platforms` with sum of its platforms.
# 3. Multiplied `RatingOverPrice` by the new column.

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

d=df["copiesSold"].groupby(df["supported_platforms"]).mean()
print(d.sort_values(ascending=False))

li=[]
dict2={"windows":10,"mac":0.01,"linux":30}
for idx,platform in df["supported_platforms"].items():
    sum1=0
    platform_list = ast.literal_eval(platform)
    for j in platform_list:
        sum1+=dict2[j]

    li.append(sum1)

df["4-GameRatingWithPlatforms"]=df["3-RatingOverPrice"]*li

# # Measure Correlation with Target

li=["4-GameRatingWithPlatforms","copiesSold"]
cor=df[li].corr()
print(cor)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ## 💿 5. NameAsCopiesSold
# 
# - Encoded `name_x` column similar to `genres`.
# 
# ### 🧩 Steps:
# 1. Preprocessed the names using nlp techniques (Tokenization, StopWords Removal, Lemmatization)
# 2. Get total copiesSold for each unique token.
# 3. Replaced every row with mean of its tokens.
# 4. Divided by 10,000 to make the values smaller.

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    words = text.split()
    cleaned_words = []
    for word in words:
        word = word.strip().lower()
        word = re.sub(r'[^a-zA-Z]', '', word)
        if word and word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)
    return cleaned_words

df['name'] = df['name_x'].apply(clean_and_lemmatize)

dict12={}
for idx,name in df["name"].items():
    for j in name:
        if j not in dict12:
            dict12[j]=0
        dict12[j] += df.loc[idx, "copiesSold"]

name_means = []

for idx, names_list in df["name"].items():
    mean1 = 0
    if len(names_list) != 0:
        for name in names_list:
            mean1 += dict12[name]
        mean1 /= len(names_list)
    name_means.append(mean1)

df.drop(["name_x","name"],inplace=True,axis=1)
df["5-NameAsCopiesSold"] = name_means
df["5-NameAsCopiesSold"] +=1
df["5-NameAsCopiesSold"] /= 1e4

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# ## 💯 6. GameRatingWithNames 
# 
# - Multiplied `GameRatingWithPlatforms` feature by the `NameAsCopiesSold` feature.
# - Improves correlation from **0.584** → **0.799**.

df["6-GameRatingWithNames"] = df["5-NameAsCopiesSold"] * df["4-GameRatingWithPlatforms"]

# # Measure Correlation with Target

li=["6-GameRatingWithNames","copiesSold"]
cor=df[li].corr()
print(cor)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Measure All New Columns Correlation with Target

li=["1-GameRating","2-GameRatingWithGenres","3-RatingOverPrice","4-GameRatingWithPlatforms","5-NameAsCopiesSold","6-GameRatingWithNames","copiesSold"]
cor=df[li].corr()
print(cor)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Explore The Data again

print(data_info(df))

# # Dropping Duplicates

df.drop_duplicates(inplace=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Splitting The Data

X=df.drop("copiesSold",axis=1)
y=df["copiesSold"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Log Transformed y to fix skewness

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(y_train , bins=100, ax=axes[0, 0])
axes[0, 0].set_title('y_train Distribution')

sns.histplot(y_train_log , bins=100, ax=axes[0, 1])
axes[0, 1].set_title('y_train_log Distribution')

sns.histplot(y_test , bins=100, ax=axes[1, 0])
axes[1, 0].set_title('y_test Distribution')

sns.histplot(y_test_log, bins=100, ax=axes[1, 1])
axes[1, 1].set_title('y_test_log Distribution')

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Scaling 
# 
# - `reviewScore` using MinMaxScaler (0<= reviewScore <=100, fixed range).
# - rest using RobustScaler (less sensitive than StandardScaler to outliers).

numerical_columns=["price","1-GameRating","2-GameRatingWithGenres","4-GameRatingWithPlatforms",
                   "3-RatingOverPrice","5-NameAsCopiesSold","6-GameRatingWithNames"]

Ro_scaler = RobustScaler()
x_train[numerical_columns] = Ro_scaler.fit_transform(x_train[numerical_columns])
x_test[numerical_columns] = Ro_scaler.transform(x_test[numerical_columns])

minmax_scaler = MinMaxScaler()
x_train["reviewScore"] = minmax_scaler.fit_transform(x_train[["reviewScore"]])
x_test["reviewScore"] = minmax_scaler.transform(x_test[["reviewScore"]])

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Encoding

# - `genres` and `supported_platforms` using multiLabelBinarizer

x_train['genres'] = x_train['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
x_test['genres'] = x_test['genres'].apply(lambda x: [g.strip() for g in x.split(',')])

mlb_genres = MultiLabelBinarizer()
genres_train_encoded = pd.DataFrame(mlb_genres.fit_transform(x_train['genres']), columns=mlb_genres.classes_, index=x_train.index)
genres_test_encoded = pd.DataFrame(mlb_genres.transform(x_test['genres']), columns=mlb_genres.classes_, index=x_test.index)

x_train = x_train.drop(columns=['genres'])
x_test = x_test.drop(columns=['genres'])

x_train = pd.concat([x_train, genres_train_encoded], axis=1)
x_test = pd.concat([x_test, genres_test_encoded], axis=1)

x_train['supported_platforms'] = x_train['supported_platforms'].apply(eval)
x_test['supported_platforms'] = x_test['supported_platforms'].apply(eval)

mlb_sup = MultiLabelBinarizer()
genres_train_encoded = pd.DataFrame(mlb_sup.fit_transform(x_train['supported_platforms']), columns=mlb_sup.classes_, index=x_train.index)
genres_test_encoded = pd.DataFrame(mlb_sup.transform(x_test['supported_platforms']), columns=mlb_sup.classes_, index=x_test.index)

x_train = x_train.drop(columns=['supported_platforms'])
x_test = x_test.drop(columns=['supported_platforms'])

x_train = pd.concat([x_train, genres_train_encoded], axis=1)
x_test = pd.concat([x_test, genres_test_encoded], axis=1)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# oneHot for remaining 

converted_bool_list = ["steam_achievements", "steam_trading_cards", "workshop_support"]
for col in converted_bool_list:
    x_train[col] = x_train[col].astype("object")
    x_test[col] = x_test[col].astype("object")

one_Hot_list = ["steam_achievements", "steam_trading_cards", "workshop_support","publisherClass"]
OneHot_Encoder = OneHotEncoder(handle_unknown='ignore') 

encoded_train = OneHot_Encoder.fit_transform(x_train[one_Hot_list])
encoded_test = OneHot_Encoder.transform(x_test[one_Hot_list])

new_columns = OneHot_Encoder.get_feature_names_out(one_Hot_list)

x_train = pd.concat([
    x_train.drop(columns=one_Hot_list), 
    pd.DataFrame(encoded_train, columns=new_columns, index=x_train.index)  
], axis=1)

x_test = pd.concat([
    x_test.drop(columns=one_Hot_list),  
    pd.DataFrame(encoded_test, columns=new_columns, index=x_test.index)  
], axis=1)

print(x_train.head())

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # correlation

# binary

cols = ["steam_achievements_1", "steam_achievements_2", "steam_trading_cards_1", "steam_trading_cards_2",
        "workshop_support_1", "workshop_support_2"]

correlations = []

for col in cols:
    corr, p = pointbiserialr(x_train[col], y_train)
    correlations.append(corr)

correlation_df = pd.DataFrame({
    'Feature': cols,
    'Correlation': correlations
})

print(correlation_df)

plt.figure(figsize=(14, 8))
sns.barplot(
    data=correlation_df,
    x='Feature',
    y='Correlation',
    color='red'
)
plt.axhline(0, color='black', linestyle='--')
plt.title('Point-Biserial Correlation between Features and Target', fontsize=16)
plt.ylabel('Correlation', fontsize=14)
plt.xlabel('Feature', fontsize=14)
plt.ylim(-1, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# categorical (Anova)

li=["price","reviewScore","1-GameRating","2-GameRatingWithGenres","3-RatingOverPrice",
"4-GameRatingWithPlatforms","5-NameAsCopiesSold","6-GameRatingWithNames",
"steam_achievements_1", "steam_achievements_2", "steam_trading_cards_1", "steam_trading_cards_2",
        "workshop_support_1", "workshop_support_2"]

te=x_train.drop(columns=li,axis=1)
f_scores, p_values = f_regression(te, y_train)

correlation_df = pd.DataFrame({
    'Feature': te.columns,
    'F-Score': f_scores,
    'p-value': p_values
}).sort_values(by='F-Score', ascending=False)

correlation_df = correlation_df.sort_values(by='F-Score', ascending=False)

print(correlation_df)
    
plt.figure(figsize=(10, 6))
sns.barplot(
    data=correlation_df,
    y='Feature',
    x='F-Score',
    palette='viridis'
)
plt.title('Feature Importance based on F-Regression', fontsize=16)
plt.xlabel('F-Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


drop=[]
for idx, row in correlation_df.iterrows():
    if row["F-Score"] < 10:
        drop.append(row["Feature"])

x_train.drop(columns=drop,inplace=True)
x_test.drop(columns=drop,inplace=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# numerical correlation

li=["price","reviewScore","1-GameRating","2-GameRatingWithGenres","3-RatingOverPrice",
"4-GameRatingWithPlatforms","5-NameAsCopiesSold","6-GameRatingWithNames","copiesSold"]

train_data = pd.concat([x_train, y_train], axis=1)
cor = train_data[li].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    cor,
    annot=True,     
    fmt=".2f",       
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.7}
)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()


li=[]

for idx, feature in cor['copiesSold'].items():
    if feature < 0.08:
        li.append(idx)

x_train.drop(li,inplace=True, axis=1)
x_test.drop(li, inplace=True, axis=1)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

print(x_train.shape)

# # Model Training

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Linear regresion

lr1 = LinearRegression()
lr1.fit(x_train, y_train)

y_pred = lr1.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Original y:")
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

lr2 = LinearRegression()
lr2.fit(x_train, y_train_log)

y_pred_log = lr2.predict(x_test)

y_pred_inverse = np.expm1(y_pred_log)
y_true_inverse = np.expm1(y_test_log)

mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
r2_log = r2_score(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mse_log)
mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

print("\nLog1p-transformed y:")
print("Mean Squared Error:", mse_log)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae_log)
print("R^2 Score:", r2_log)

# # Plotting the results

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='k', alpha=0.7)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Original y)')
plt.grid(True)

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, color='green', edgecolor='k', alpha=0.7)

plt.plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values (Log Inverse)')
plt.ylabel('Predicted Values (Log Inverse)')
plt.title('Actual vs Predicted Values (Log-transformed y)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Ridge Regression

model1R = Ridge(
    alpha=1,
    fit_intercept=True,
    solver='auto',
    random_state=42
)
model1R.fit(x_train, y_train)

y_pred = model1R.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Original y:")
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

model2R = Ridge(
    alpha=1,
    fit_intercept=True,
    solver='auto',
    random_state=42
)

model2R.fit(x_train, y_train_log)

y_pred_log = model2R.predict(x_test)

y_pred_inverse = np.expm1(y_pred_log)
y_true_inverse = np.expm1(y_test_log)

mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
r2_log = r2_score(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mse_log)
mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

print("\nLog1p-transformed y:")
print("Mean Squared Error:", mse_log)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae_log)
print("R^2 Score:", r2_log)

# # Plotting The results

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='k', alpha=0.7)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Original y - Ridge)')
plt.grid(True)

plt.subplot(1, 2, 2) 
sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, color='green', edgecolor='k', alpha=0.7)

plt.plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values (Log Inverse)')
plt.ylabel('Predicted Values (Log Inverse)')
plt.title('Actual vs Predicted Values (Log-transformed y - Ridge)')
plt.grid(True)

plt.tight_layout() 
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # Lasso Regression

model1L = Lasso(
    alpha=0.0001,
    fit_intercept=True,
    max_iter=1000,
    tol=0.0001,
    random_state=42
)
model1L.fit(x_train, y_train)

y_pred = model1L.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Original y:")
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

model2L = Lasso(
    alpha=0.0001,
    fit_intercept=True,
    max_iter=1000,
    tol=0.0001,
    random_state=42
)

model2L.fit(x_train, y_train_log)

y_pred_log = model2L.predict(x_test)

y_pred_inverse = np.expm1(y_pred_log)
y_true_inverse = np.expm1(y_test_log)

mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
r2_log = r2_score(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mse_log)
mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

print("\nLog1p-transformed y:")
print("Mean Squared Error:", mse_log)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae_log)
print("R^2 Score:", r2_log)

# # Plotting The results

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='k', alpha=0.7)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Original y - Lasso)')
plt.grid(True)

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, color='green', edgecolor='k', alpha=0.7)

plt.plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values (Log Inverse)')
plt.ylabel('Predicted Values (Log Inverse)')
plt.title('Actual vs Predicted Values (Log-transformed y - Lasso)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # RandomForest & GridSearch

# rf = RandomForestRegressor(random_state=42)

# param_grid = {
#     'n_estimators': [50, 100, 200],  
#     'max_depth': [None, 10, 20, 30],  
#     'min_samples_split': [2, 5, 10],  
#     'min_samples_leaf': [1, 2, 4],    
#     'max_features': ['auto', 'sqrt', 'log2']  
# }

# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring='neg_mean_squared_error',  
#     n_jobs=-1,  
#     verbose=1
# )

# grid_search.fit(x_train, y_train)
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)
# print("\nTest Set Evaluation using Best Model from GridSearchCV:")
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)


# Fitting 5 folds for each of 324 candidates, totalling 1620 fits
# Best Hyperparameters from GridSearchCV: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
# 
# Test Set Evaluation using Best Model from GridSearchCV:
# Mean Squared Error: 2.0504432758918125
# R^2 Score: 0.7914442767029305

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # RandomForest w/ best parameters from GridSearch

rf1 = RandomForestRegressor(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=200
)
rf1.fit(x_train, y_train)

y_pred = rf1.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Original y:")
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

rf_log = RandomForestRegressor(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=200
)

rf_log.fit(x_train, y_train_log)

y_pred_log = rf_log.predict(x_test)

y_pred_inverse = np.expm1(y_pred_log)
y_true_inverse = np.expm1(y_test_log)

mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
r2_log = r2_score(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mse_log)
mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

print("\nLog1p-transformed y:")
print("Mean Squared Error:", mse_log)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae_log)
print("R^2 Score:", r2_log)


# # Plotting The results


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='k', alpha=0.7)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Original y - Random Forest)')
plt.grid(True)

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, color='green', edgecolor='k', alpha=0.7)

plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot (Original y - Random Forest)')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, color='blue', edgecolor='k', alpha=0.7)

plt.plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], color='red', linestyle='--', linewidth=2)

plt.xlabel('Actual Values (Log Inverse)')
plt.ylabel('Predicted Values (Log Inverse)')
plt.title('Actual vs Predicted Values (Log-transformed y - Random Forest)')
plt.grid(True)

plt.subplot(1, 2, 2)
residuals_log = y_true_inverse - y_pred_inverse
sns.scatterplot(x=y_pred_inverse, y=residuals_log, color='green', edgecolor='k', alpha=0.7)

plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values (Log Inverse)')
plt.ylabel('Residuals')
plt.title('Residuals Plot (Log-transformed y - Random Forest)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # SVM & PCA

# pca = PCA(n_components=None)  

# x_trainP = pca.fit_transform(x_train)
# x_testP = pca.transform(x_test)

# svm1 = SVR(kernel='rbf', C=10.0, epsilon=0.01)

# svm1.fit(x_trainP, y_train)
# y_pred = svm1.predict(x_testP)

# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Original y:")
# print("Mean Squared Error:", mse)
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# print("Mean Absolute Error:", mae)
# print("R^2 Score:", r2)

# svm_log = SVR(kernel='rbf', C=10.0, epsilon=0.01)

# svm_log.fit(x_trainP, y_train_log)
# y_pred_log = svm_log.predict(x_testP)

# y_pred_inverse = np.expm1(y_pred_log)
# y_true_inverse = np.expm1(y_test_log)

# mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
# r2_log = r2_score(y_true_inverse, y_pred_inverse)
# rmse = np.sqrt(mse_log)
# mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

# print("\nLog1p-transformed y:")
# print("Mean Squared Error:", mse_log)
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# print("Mean Absolute Error:", mae_log)
# print("R^2 Score:", r2_log)


# Original y:
# Mean Squared Error: 4732765618111.92
# Root Mean Squared Error (RMSE): 2175492.04
# Mean Absolute Error: 96101.23775416201
# R^2 Score: -0.0019142457110299382
# 
# Log1p-transformed y:
# Mean Squared Error: 1.4015269905505807e+39
# Root Mean Squared Error (RMSE): 37436973576273240064.00
# Mean Absolute Error: 3.180056189282022e+17
# R^2 Score: -2.9669964052463e+26

# # Plotting The results

# fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# fig.suptitle('SVM with PCA - Original y', fontsize=18)

# sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], color='dodgerblue', edgecolor='w', alpha=0.7)
# axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
# axes[0].set_xlabel('Actual Values')
# axes[0].set_ylabel('Predicted Values')
# axes[0].set_title('Actual vs Predicted')
# axes[0].grid(True)

# residuals = y_test - y_pred
# sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], color='green', edgecolor='w', alpha=0.7)
# axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
# axes[1].set_xlabel('Predicted Values')
# axes[1].set_ylabel('Residuals')
# # axes[1].set_title('Residuals Plot')
# axes[1].grid(True)

# explained_variance_ratio = pca.explained_variance_ratio_ * 100
# components = np.arange(1, len(explained_variance_ratio) + 1)
# sns.barplot(x=components, y=explained_variance_ratio, ax=axes[2], palette="viridis")
# axes[2].set_title('PCA Explained Variance (%)')
# axes[2].set_xlabel('Principal Component')
# axes[2].set_ylabel('Variance Explained (%)')
# axes[2].set_xticks(components)
# axes[2].tick_params(axis='x', rotation=90)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# fig.suptitle('SVM with PCA - Log1p(y)', fontsize=18)

# sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, ax=axes[0], color='dodgerblue', edgecolor='w', alpha=0.7)
# axes[0].plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], 'r--', linewidth=2)
# axes[0].set_xlabel('Actual Values (Inverse Log)')
# axes[0].set_ylabel('Predicted Values (Inverse Log)')
# axes[0].set_title('Actual vs Predicted (Log Inverse)')
# axes[0].grid(True)

# residuals_log = y_true_inverse - y_pred_inverse
# sns.scatterplot(x=y_pred_inverse, y=residuals_log, ax=axes[1], color='green', edgecolor='w', alpha=0.7)
# axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
# axes[1].set_xlabel('Predicted Values (Inverse Log)')
# axes[1].set_ylabel('Residuals')
# axes[1].set_title('Residuals Plot (Log Inverse)')
# axes[1].grid(True)

# sns.barplot(x=components, y=explained_variance_ratio, ax=axes[2], palette="viridis")
# axes[2].set_title('PCA Explained Variance (%)')
# axes[2].set_xlabel('Principal Component')
# axes[2].set_ylabel('Variance Explained (%)')
# axes[2].set_xticks(components)
# axes[2].tick_params(axis='x', rotation=90)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # XGBOOST & GridSearch

# xgb = XGBRegressor(
#     objective='reg:squarederror',
#     random_state=42,
#     n_jobs=-1
# )

# param_grid = {
#     'n_estimators': [100, 200, 300],             
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],     
#     'max_depth': [3, 5, 7, 9],                  
#     'subsample': [0.6, 0.8, 1.0],                 
#     'colsample_bytree': [0.6, 0.8, 1.0]          
# }

# grid_search = GridSearchCV(
#     estimator=xgb,
#     param_grid=param_grid,
#     cv=5,
#     scoring='r2',
#     n_jobs=-1,
#     verbose=1
# )

# grid_search.fit(x_train, y_train)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Best Hyperparameters:", grid_search.best_params_)
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)


# Fitting 5 folds for each of 432 candidates, totalling 2160 fits
# Best Hyperparameters: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 300, 'subsample': 0.8}
# Mean Squared Error: 1.8937102424468064
# R^2 Score: 0.8073859862537348

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # XGBOOST w/ Best Parameters from Grid Search

model1 = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=9,
    subsample=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model1.fit(x_train, y_train)

y_pred = model1.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Original y:")
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

model_log = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=9,
    subsample=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model_log.fit(x_train, y_train_log)

y_pred_log = model_log.predict(x_test)

y_pred_inverse = np.expm1(y_pred_log)
y_true_inverse = np.expm1(y_test_log)

mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
r2_log = r2_score(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mse_log)
mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

print("\nLog1p-transformed y:")
print("Mean Squared Error:", mse_log)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae_log)
print("R^2 Score:", r2_log)


# # Plotting The results

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Evaluation (Original y)', fontsize=16)

sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='w', alpha=0.7, ax=axes[0])
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('Actual vs Predicted')
axes[0].grid(True)

residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, color='green', edgecolor='w', alpha=0.7, ax=axes[1])
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot')
axes[1].grid(True)

xgb.plot_importance(model1, importance_type='weight', max_num_features=10, height=0.5, ax=axes[2])
axes[2].set_title('Feature Importance')

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Evaluation (Log-transformed y)', fontsize=16)

sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, color='blue', edgecolor='w', alpha=0.7, ax=axes[0])
axes[0].plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Actual Values (Log Inverse)')
axes[0].set_ylabel('Predicted Values (Log Inverse)')
axes[0].set_title('Actual vs Predicted (Log-transformed y)')
axes[0].grid(True)

residuals_log = y_true_inverse - y_pred_inverse
sns.scatterplot(x=y_pred_inverse, y=residuals_log, color='green', edgecolor='w', alpha=0.7, ax=axes[1])
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values (Log Inverse)')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot (Log-transformed y)')
axes[1].grid(True)

xgb.plot_importance(model_log, importance_type='weight', max_num_features=10, height=0.5, ax=axes[2])
axes[2].set_title('Feature Importance (Log-transformed y)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # LGB &  Grid Search

# import lightgbm as lgb
# model = lgb.LGBMRegressor(random_state=42)

# param_grid = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.05, 0.1],
#     'num_leaves': [31, 50],
#     'max_depth': [-1, 10]
# }

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
# grid_search.fit(x_train, y_train)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Best LightGBM Params:", grid_search.best_params_)
# print(f"Test MSE: {mse:.4f}")
# print(f"Test R² Score: {r2:.4f}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # LGB w/ Best Parameters from Grid Search

model1L = lgb.LGBMRegressor(
    n_estimators=270,
    learning_rate=0.06,
    num_leaves=50,
    max_depth=10,
    random_state=42,
    min_child_samples=20,
)

model1L.fit(x_train, y_train)

y_pred = model1L.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Original y:")
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

model2L = lgb.LGBMRegressor(
    n_estimators=270,
    learning_rate=0.06,
    num_leaves=50,
    max_depth=10,
    random_state=42,
    min_child_samples=20,
    verbosity=-1
)

model2L.fit(x_train, y_train_log)

y_pred_log = model2L.predict(x_test)

y_pred_inverse = np.expm1(y_pred_log)
y_true_inverse = np.expm1(y_test_log)

mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
r2_log = r2_score(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mse_log)
mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

print("\nLog1p-transformed y:")
print("Mean Squared Error:", mse_log)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae_log)
print("R^2 Score:", r2_log)


# # Plotting The results

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Model 1L: LGBM on Original y', fontsize=18)

sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], color='dodgerblue', edgecolor='w', alpha=0.7)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('Actual vs Predicted')
axes[0].grid(True)

residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], color='green', edgecolor='w', alpha=0.7)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot')
axes[1].grid(True)

lgb.plot_importance(model1L, importance_type='gain', max_num_features=10, height=0.5, ax=axes[2])
axes[2].set_title('Feature Importance (Gain)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Model 2L: LGBM on Log1p(y)', fontsize=18)

sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, ax=axes[0], color='dodgerblue', edgecolor='w', alpha=0.7)
axes[0].plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Values (Inverse Log)')
axes[0].set_ylabel('Predicted Values (Inverse Log)')
axes[0].set_title('Actual vs Predicted (Log Inverse)')
axes[0].grid(True)

residuals_log = y_true_inverse - y_pred_inverse
sns.scatterplot(x=y_pred_inverse, y=residuals_log, ax=axes[1], color='green', edgecolor='w', alpha=0.7)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values (Inverse Log)')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot (Log Inverse)')
axes[1].grid(True)

lgb.plot_importance(model2L, importance_type='gain', max_num_features=10, height=0.5, ax=axes[2])
axes[2].set_title('Feature Importance (Gain)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # CatBoost & Grid Search

# model = CatBoostRegressor(verbose=0, random_state=42)

# param_grid = {
#     'iterations': [200, 300],
#     'learning_rate': [0.03, 0.1],
#     'depth': [6, 10]
# }

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
# grid_search.fit(x_train, y_train)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Best CatBoost Params:", grid_search.best_params_)
# print(f"Test MSE: {mse:.4f}")
# print(f"Test R² Score: {r2:.4f}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # CatBoost w/ Best Parameters from Grid Search

model1C = CatBoostRegressor(
    iterations=280,
    learning_rate=0.06,
    depth=11,
    verbose=0,
    random_state=42
)

model1C.fit(x_train, y_train)

y_pred = model1C.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Original y:")
print("Mean Squared Error:", mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

model2C = CatBoostRegressor(
    iterations=280,
    learning_rate=0.06,
    depth=11,
    verbose=0,
    random_state=42
)
model2C.fit(x_train, y_train_log)

y_pred_log = model2C.predict(x_test)

y_pred_inverse = np.expm1(y_pred_log)
y_true_inverse = np.expm1(y_test_log)

mse_log = mean_squared_error(y_true_inverse, y_pred_inverse)
r2_log = r2_score(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mse_log)
mae_log= mean_absolute_error(y_true_inverse, y_pred_inverse)

print("\nLog1p-transformed y:")
print("Mean Squared Error:", mse_log)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Mean Absolute Error:", mae_log)
print("R^2 Score:", r2_log)


x_train.head()


# # Plotting The results

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Model 1C: CatBoost on Original y', fontsize=18)

sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], color='dodgerblue', edgecolor='w', alpha=0.7)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('Actual vs Predicted')
axes[0].grid(True)

residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], color='green', edgecolor='w', alpha=0.7)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot')
axes[1].grid(True)

feature_importances = model1C.get_feature_importance()
features = x_train.columns
sorted_idx = feature_importances.argsort()[::-1][:10]

sns.barplot(x=feature_importances[sorted_idx], y=features[sorted_idx], ax=axes[2], palette="viridis")
axes[2].set_title('Feature Importance (CatBoost)')
axes[2].set_xlabel('Importance')
axes[2].set_ylabel('Features')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Model 2C: CatBoost on Log1p(y)', fontsize=18)

sns.scatterplot(x=y_true_inverse, y=y_pred_inverse, ax=axes[0], color='dodgerblue', edgecolor='w', alpha=0.7)
axes[0].plot([y_true_inverse.min(), y_true_inverse.max()], [y_true_inverse.min(), y_true_inverse.max()], 'r--', linewidth=2)
axes[0].set_xlabel('Actual Values (Inverse Log)')
axes[0].set_ylabel('Predicted Values (Inverse Log)')
axes[0].set_title('Actual vs Predicted (Log Inverse)')
axes[0].grid(True)

residuals_log = y_true_inverse - y_pred_inverse
sns.scatterplot(x=y_pred_inverse, y=residuals_log, ax=axes[1], color='green', edgecolor='w', alpha=0.7)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values (Inverse Log)')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot (Log Inverse)')
axes[1].grid(True)

feature_importances = model2C.get_feature_importance()
features = x_train.columns
sorted_idx = feature_importances.argsort()[::-1][:10]

sns.barplot(x=feature_importances[sorted_idx], y=features[sorted_idx], ax=axes[2], palette="viridis")
axes[2].set_title('Feature Importance (CatBoost)')
axes[2].set_xlabel('Importance')
axes[2].set_ylabel('Features')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# # After Comparison of Models in terms of (rmse,mae,r2-score) catboost is the best model to use in deployment with original target

model1C.save_model('model1C.cbm')

joblib.dump(Ro_scaler, 'robust_scaler.pkl')
joblib.dump(minmax_scaler, 'minmax_scaler.pkl')

joblib.dump(OneHot_Encoder, 'onehot_encoder.pkl')
joblib.dump(mlb_genres, 'mlb_genres_encoder.pkl')

joblib.dump(dict1, 'dict1_genres.pkl')
joblib.dump(dict2, 'dict2_platforms.pkl')
joblib.dump(dict12, 'dict12_names.pkl')
joblib.dump(mlb_sup, 'mlb_platforms_encoder.pkl')

# --------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------