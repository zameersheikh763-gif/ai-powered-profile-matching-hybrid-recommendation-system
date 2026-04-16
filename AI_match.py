from flask import Flask, request, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)


# LOAD DATA

users = pd.read_csv(r"C:\Users\ADMIN\Desktop\.vscode\python\UNLOX_ACADEMY\NLP_PROJECT\users (1).csv")
feedback = pd.read_csv(r"C:\Users\ADMIN\Desktop\.vscode\python\UNLOX_ACADEMY\NLP_PROJECT\feedback (1).csv")


# NLP

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = str(text).lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

users["combined_text"] = users["professional_summary"] + " " + users["about_me"]
users["clean_text"] = users["combined_text"].apply(preprocess)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(users["clean_text"])


# FEATURE FUNCTIONS

def get_text_similarity(u1, u2):
    i1 = users[users["user_id"] == u1].index[0]
    i2 = users[users["user_id"] == u2].index[0]
    return cosine_similarity(tfidf_matrix[i1], tfidf_matrix[i2])[0][0]

def get_mbti_score(m1, m2):
    return 1.0 if m1[0] == m2[0] else 0.5

def get_location_score(l1, l2):
    return 1.0 if l1 == l2 else 0.3

def get_profession_score(p1, p2):
    return 1.0 if p1 == p2 else 0.3

def get_interest_score(i1, i2):
    s1 = set(str(i1).lower().split(","))
    s2 = set(str(i2).lower().split(","))
    return len(s1 & s2) / max(len(s1 | s2), 1)


# SIMPLE RULE (BEFORE)

def simple_score(u1, u2):
    d1 = users[users["user_id"] == u1].iloc[0]
    d2 = users[users["user_id"] == u2].iloc[0]

    score = 0
    if d1["location"] == d2["location"]:
        score += 1
    if d1["profession"] == d2["profession"]:
        score += 1
    if d1["mbti"][0] == d2["mbti"][0]:
        score += 1

    return 1 if score >= 2 else 0
# TRAIN MODEL

def train_model():
    X, y = [], []

    for _, r in feedback.iterrows():
        u1, u2 = r["user_id"], r["matched_user_id"]

        d1 = users[users["user_id"] == u1].iloc[0]
        d2 = users[users["user_id"] == u2].iloc[0]

        X.append([
            get_text_similarity(u1, u2),
            get_mbti_score(d1["mbti"], d2["mbti"]),
            get_location_score(d1["location"], d2["location"]),
            get_profession_score(d1["profession"], d2["profession"]),
            get_interest_score(d1["interests"], d2["interests"])
        ])
        y.append(r["action"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model


# ACCURACY

def calculate_before_accuracy():
    correct = 0
    for _, r in feedback.iterrows():
        pred = simple_score(r["user_id"], r["matched_user_id"])
        if pred == r["action"]:
            correct += 1
    return round(correct / len(feedback) * 100, 2)

def calculate_after_accuracy(model):
    correct = 0
    for _, r in feedback.iterrows():
        u1, u2 = r["user_id"], r["matched_user_id"]

        d1 = users[users["user_id"] == u1].iloc[0]
        d2 = users[users["user_id"] == u2].iloc[0]

        features = [[
            get_text_similarity(u1, u2),
            get_mbti_score(d1["mbti"], d2["mbti"]),
            get_location_score(d1["location"], d2["location"]),
            get_profession_score(d1["profession"], d2["profession"]),
            get_interest_score(d1["interests"], d2["interests"])
        ]]

        pred = model.predict(features)[0]

        if pred == r["action"]:
            correct += 1

    return round(correct / len(feedback) * 100, 2)

# RECOMMEND

def recommend(uid, model):
    res = []

    for u in users["user_id"]:
        if u != uid:
            d1 = users[users["user_id"] == uid].iloc[0]
            d2 = users[users["user_id"] == u].iloc[0]

            features = [[
                get_text_similarity(uid, u),
                get_mbti_score(d1["mbti"], d2["mbti"]),
                get_location_score(d1["location"], d2["location"]),
                get_profession_score(d1["profession"], d2["profession"]),
                get_interest_score(d1["interests"], d2["interests"])
            ]]

            prob = model.predict_proba(features)[0][1]

            res.append({
                "name": d2["name"],
                "id": u,
                "profession": d2["profession"],
                "score": round(float(prob * 100), 2)
            })

    return sorted(res, key=lambda x: x["score"], reverse=True)[:5]


# HTML (DARK UI)

HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
body {background:#121212;color:white;font-family:Arial;}
.container {width:650px;margin:50px auto;background:#1e1e1e;padding:30px;border-radius:10px;text-align:center;}
.card {background:#2c2c2c;padding:10px;margin:10px;border-radius:8px;}
button {background:#00adb5;color:white;padding:10px;border:none;}
</style>
</head>
<body>
<div class="container">
<h2>Profile Matching System</h2>

<form method="POST">
<input name="user_id" value="U001">
<button>Run</button>
</form>

{% if results %}
<h3>Before: {{before}}%</h3>
<h3>After: {{after}}%</h3>

{% for m in results %}
<div class="card">
<b>{{m.name}}</b> ({{m.id}})<br>
{{m.profession}}<br>
Match: {{m.score}}%
</div>
{% endfor %}

<img src="data:image/png;base64,{{graph}}" width="400">
{% endif %}

</div>
</body>
</html>
"""


# ROUTE

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":

        before = calculate_before_accuracy()

        model = train_model()
        after = calculate_after_accuracy(model)

        results = recommend(request.form["user_id"], model)

        img = io.BytesIO()
        plt.figure()
        plt.bar(["Before","After"], [before, after])
        plt.savefig(img, format='png')  
        plt.close()

        img.seek(0)
        graph = base64.b64encode(img.getvalue()).decode()

        return render_template_string(
            HTML,
            results=results,
            before=before,
            after=after,
            graph=graph
        )

    return render_template_string(HTML)

if __name__ == "__main__":
    app.run(debug=True)
