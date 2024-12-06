import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Import packages
import pandas as pd
import numpy as np
from collections import Counter

#For visualization
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

#For User-Collaborative Filtering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

#For Encoding
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

#warning
import warnings

pd.options.display.float_format = '{:,.6f}'.format


warnings.filterwarnings("ignore") # To prevent kernel from showing any warning


df = pd.read_csv("Food_Recipe.csv")

df.head()


df.info()

df.isnull().sum()

new_df = df[['name','cuisine','course','diet', 'ingredients_name', 'prep_time (in mins)','cook_time (in mins)']]
#Define all categorical columns
cat_cols = (new_df.dtypes[new_df.dtypes == 'object']).index

#Convert all object columns to title case
for col in cat_cols:
    new_df[col] = new_df[col].str.title()

#Display first 5 rows of new dataframe
new_df.head()

df['ingredients_name'][0]

new_df.dropna(inplace=True)

new_df.isnull().sum()

new_df.drop_duplicates(inplace=True)

new_df.duplicated().sum()

new_df.nunique()

new_df['cuisine'].value_counts()


new_df['course'].value_counts()


new_df['diet'].value_counts()


new_df.describe()


new_df[new_df['prep_time (in mins)'] > 1440]


new_df['prep_time'] = np.where(new_df['prep_time (in mins)'] == 0,'no prep',
                                np.where(new_df['prep_time (in mins)'] <= 30,'under 30 mins',
                                np.where(new_df['prep_time (in mins)'] <= 60,'under 1 hour',
                                np.where(new_df['prep_time (in mins)'] <= 300,'under 5 hours',
                                np.where(new_df['prep_time (in mins)'] > 300, 'more than 5 hours',
                                np.where(new_df['prep_time (in mins)'] > 1440, 'more than 1 day','na'))))))


new_df['cook_time'] = np.where(new_df['cook_time (in mins)'] == 0,'no cook time',
                                np.where(new_df['cook_time (in mins)'] <= 30,'under 30 mins',
                                np.where(new_df['prep_time (in mins)'] <= 60,'under 1 hour',
                                np.where(new_df['prep_time (in mins)'] <= 300,'under 5 hours',
                                np.where(new_df['prep_time (in mins)'] > 300, 'more than 5 hours', 'more than 1 day')))))

#Drop `prep_time (in mins)` and `cook_time (in mins)` columns
new_df.drop(['prep_time (in mins)', 'cook_time (in mins)'], axis=1, inplace=True)
#Double check dataframe
new_df.head()

#Prep data to contain only numeric values
cuisine_num = pd.get_dummies(new_df['cuisine']).astype(int)

course_num = pd.get_dummies(new_df['course']).astype(int)

diet_num = pd.get_dummies(new_df['diet']).astype(int)

prep_num = pd.get_dummies(new_df['prep_time']).astype(int)

cook_num = pd.get_dummies(new_df['cook_time']).astype(int)

#Combine them into single dataframe
X = pd.concat([cuisine_num,
               course_num,
               diet_num,
               prep_num,
               cook_num], axis= 1).set_index(new_df['name'])
X.head()

# Ensure ingredients are strings
df['ingredients_name'] = df['ingredients_name'].astype(str)

# Example: Convert to space-separated lists of ingredients
df['ingredients_tokenized'] = df['ingredients_name'].apply(lambda x: x.replace(", ", " "))


from sklearn.feature_extraction.text import CountVectorizer

# Create binary vectors for ingredients
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary=True)
ingredient_vectors = vectorizer.fit_transform(df['ingredients_tokenized'])

# Convert sparse matrix to dense format for easier manipulation
ingredient_vectors = ingredient_vectors.toarray()


def recommend_recipes_by_ingredients(target_ingredients, n, df, ingredient_vectors, vectorizer):
    """
    Recommend recipes based on ingredient similarity.

    Parameters:
    - target_ingredients (list): List of ingredients to search for similar recipes.
    - n (int): Number of recipes to return.
    - df (DataFrame): The original dataset.
    - ingredient_vectors (array): Binary ingredient matrix.
    - vectorizer (CountVectorizer): Fitted vectorizer for ingredients.

    Returns:
    - DataFrame: Top `n` similar recipes.
    """
    # Convert target ingredients into a vector
    target_vector = vectorizer.transform([" ".join(target_ingredients)]).toarray()

    # Compute cosine similarity
    similarity_scores = cosine_similarity(target_vector, ingredient_vectors)[0]

    # Add similarity scores to DataFrame
    df['similarity'] = similarity_scores

    # Sort by similarity and return top n recipes
    recommended = df.sort_values(by='similarity', ascending=False).head(n)
    return recommended[['name', 'ingredients_name', 'similarity', 'instructions']]

def recommend_recipes_by_ingredients_call(from_ui_ingredients, number):
   return  recommend_recipes_by_ingredients(from_ui_ingredients, number, df, ingredient_vectors, vectorizer)




