from flask import Flask, render_template, request
from model import recipe_logic
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get form data
    ingredients = request.form.get("ingredients", "").split(",")  # Convert to a list

    # Get recommendations
    recipes = recipe_logic.recommend_recipes_by_ingredients_call(ingredients, 6)
    print(recipes)

    # Pass results to the template
    return render_template("results.html", recipes=recipes.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
