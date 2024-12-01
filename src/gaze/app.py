from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///local.db")
db = SQLAlchemy(app)
migrate = Migrate(app, db)


@app.route("/predict", methods=["POST"])
def predict():
    # Placeholder for model logic
    return jsonify({"prediction": "placeholder"})


if __name__ == "__main__":
    app.run(debug=True)
