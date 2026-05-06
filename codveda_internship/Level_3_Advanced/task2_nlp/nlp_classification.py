"""
Codveda Technology Internship
Level 3 - Task 2: Natural Language Processing (NLP) – Text Classification
Author: Peter Molepo
Description: Classify customer reviews as Positive or Negative using NLP techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Optional: use nltk stopwords if available, else fallback
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOP_WORDS = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    USE_NLTK = True
except Exception:
    USE_NLTK = False
    STOP_WORDS = {"i", "me", "my", "the", "a", "an", "is", "it", "in", "on",
                  "and", "or", "but", "to", "of", "for", "this", "that", "was",
                  "are", "be", "with", "at", "by", "from", "not", "no"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titleweight": "bold",
})


# ─────────────────────────────────────────────
#  1. GENERATE REVIEW DATASET
# ─────────────────────────────────────────────

POSITIVE_REVIEWS = [
    "This product is absolutely fantastic, I love it so much",
    "Amazing quality and super fast delivery, very happy",
    "Excellent service, the team was professional and helpful",
    "Best purchase I have made this year, highly recommend",
    "Outstanding performance, exceeded all my expectations",
    "The product works perfectly and looks great too",
    "Very satisfied with my purchase, will buy again",
    "Incredible value for money, top quality",
    "Superb experience from start to finish",
    "Five stars, everything was smooth and easy",
    "Really happy with this, does exactly what it says",
    "Brilliant product, arrived quickly and well packaged",
    "Love this item, it has made my life so much easier",
    "Great customer support and fast response times",
    "Perfect fit and exactly as described, wonderful",
    "So pleased with this purchase, works like a charm",
    "Top notch quality and great price, very impressed",
    "Highly satisfied, this is exactly what I needed",
    "Would definitely recommend to friends and family",
    "Fantastic value, well built and easy to use",
    "Smooth ordering process and quick delivery",
    "Product exceeded my expectations, very impressed",
    "Great quality item, solid and well made",
    "Happy with everything, great overall experience",
    "Loved it from day one, works without any issues",
]

NEGATIVE_REVIEWS = [
    "Terrible product, broke after just two days of use",
    "Very disappointed, nothing like what was advertised",
    "Poor quality and extremely slow delivery, not happy",
    "Worst purchase ever, complete waste of money",
    "Product stopped working after one week, total failure",
    "Customer service was rude and completely unhelpful",
    "Arrived damaged and the return process is a nightmare",
    "Do not buy this, it is a total scam and waste",
    "Extremely poor quality, fell apart immediately",
    "Not worth the money at all, very disappointed",
    "Packaging was destroyed and item was broken inside",
    "This product is garbage, does not work as described",
    "Worst experience ever, will never shop here again",
    "Complete disaster, took weeks to arrive and broken",
    "Awful quality, cheaply made and not durable at all",
    "Terrible customer experience, no one responded to me",
    "Received the wrong item and support was useless",
    "Broke on first use, very poor build quality",
    "Highly disappointed with both product and service",
    "Never again, this was a total waste of time and money",
    "Item was defective and replacement took too long",
    "The worst product I have ever bought in my life",
    "Misleading description, product is nothing like shown",
    "Support team ignored my complaint for two weeks",
    "Very unhappy, this failed to meet any expectations",
]


def create_dataset():
    """Generate a balanced synthetic text classification dataset."""
    np.random.seed(42)

    reviews = []
    labels = []

    # Expand dataset with slight variations
    for _ in range(12):
        for review in POSITIVE_REVIEWS:
            words = review.split()
            np.random.shuffle(words)
            variation = " ".join(words[:max(5, len(words) - np.random.randint(0, 3))])
            reviews.append(variation)
            labels.append(1)

        for review in NEGATIVE_REVIEWS:
            words = review.split()
            np.random.shuffle(words)
            variation = " ".join(words[:max(5, len(words) - np.random.randint(0, 3))])
            reviews.append(variation)
            labels.append(0)

    df = pd.DataFrame({"Review": reviews, "Sentiment": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("customer_reviews.csv", index=False)

    print(f"\n  Dataset size     : {len(df)} reviews")
    print(f"  Positive reviews : {df['Sentiment'].sum()}")
    print(f"  Negative reviews : {(df['Sentiment'] == 0).sum()}")

    return df


# ─────────────────────────────────────────────
#  2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def clean_text(text):
    """Tokenise, lowercase, remove punctuation, stopwords, and lemmatize."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)                 # remove non-alpha
    text = re.sub(r"\s+", " ", text).strip()             # normalise whitespace

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if USE_NLTK:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess(df):
    print("\n" + "=" * 60)
    print("  STEP 2: TEXT PREPROCESSING")
    print("=" * 60)

    df = df.copy()
    df["CleanedReview"] = df["Review"].apply(clean_text)

    print(f"\n  Sample before cleaning : {df['Review'].iloc[0]}")
    print(f"  Sample after cleaning  : {df['CleanedReview'].iloc[0]}")

    X = df["CleanedReview"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,    # apply log normalization
        min_df=2,
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    print(f"\n  TF-IDF vocabulary size : {len(tfidf.vocabulary_):,}")
    print(f"  Training matrix shape  : {X_train_vec.shape}")
    print(f"  Test matrix shape      : {X_test_vec.shape}")

    return X_train_vec, X_test_vec, y_train, y_test, tfidf, df


# ─────────────────────────────────────────────
#  3. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def eval_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Predictions": y_pred,
        "Model_Obj": model,
    }


def train_all(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("  STEP 3: MODEL TRAINING")
    print("=" * 60)

    results = []

    results.append(eval_model("Logistic Regression",
                               LogisticRegression(max_iter=1000, C=1.0, random_state=42),
                               X_train, X_test, y_train, y_test))

    results.append(eval_model("Naive Bayes",
                               MultinomialNB(alpha=0.1),
                               X_train, X_test, y_train, y_test))

    results.append(eval_model("Linear SVM",
                               LinearSVC(max_iter=2000, C=1.0, random_state=42),
                               X_train, X_test, y_train, y_test))

    results.append(eval_model("Random Forest",
                               RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                               X_train, X_test, y_train, y_test))

    print(f"\n  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['Model']:<22} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} "
              f"{r['Recall']:>8.4f} {r['F1']:>8.4f}")

    return results


# ─────────────────────────────────────────────
#  4. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_results(results, y_test, tfidf, df):
    best = max(results, key=lambda x: x["F1"])

    # ── Plot 1: Metrics Comparison ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("NLP Classifier Comparison – Sentiment Analysis", fontsize=14, fontweight="bold")

    names = [r["Model"] for r in results]
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(names))
    width = 0.2
    colors = ["#3498DB", "#E74C3C", "#27AE60", "#F39C12"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        axes[0].bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("All Metrics per Model")
    axes[0].legend()

    f1_vals = [r["F1"] for r in results]
    bar_colors = ["#E74C3C" if n == best["Model"] else "#AED6F1" for n in names]
    bars = axes[1].bar(names, f1_vals, color=bar_colors)
    axes[1].set_title("F1 Score (best in red)")
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis="x", rotation=15)
    for bar, v in zip(bars, f1_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("plot_01_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: plot_01_model_comparison.png")

    # ── Plot 2: Confusion Matrix ──────────────────────────────
    cm = confusion_matrix(y_test, best["Predictions"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {best['Model']}")
    plt.tight_layout()
    plt.savefig("plot_02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_02_confusion_matrix.png")

    # ── Plot 3: Top TF-IDF words per class ───────────────────
    feature_names = np.array(tfidf.get_feature_names_out())
    model = best["Model_Obj"]

    if hasattr(model, "coef_"):
        coef = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
        top_pos = feature_names[np.argsort(coef)[-20:]]
        top_neg = feature_names[np.argsort(coef)[:20]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Top 20 Predictive Words – {best['Model']}", fontsize=14, fontweight="bold")

        axes[0].barh(top_pos, sorted(coef)[-20:], color="#27AE60")
        axes[0].set_title("Positive Sentiment Words")
        axes[0].set_xlabel("Coefficient Weight")

        axes[1].barh(top_neg, sorted(coef)[:20], color="#E74C3C")
        axes[1].set_title("Negative Sentiment Words")
        axes[1].set_xlabel("Coefficient Weight")

        plt.tight_layout()
        plt.savefig("plot_03_top_words.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: plot_03_top_words.png")

    print(f"\n  Best model: {best['Model']}  (F1 = {best['F1']:.4f})")
    print(f"\n  Classification Report – {best['Model']}:")
    print(classification_report(y_test, best["Predictions"],
                                 target_names=["Negative", "Positive"]))


# ─────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CODVEDA INTERNSHIP | LEVEL 3 – TASK 2")
    print("  NLP – Sentiment Classification")
    print("=" * 60)

    df = create_dataset()
    X_train, X_test, y_train, y_test, tfidf, df_clean = preprocess(df)
    results = train_all(X_train, X_test, y_train, y_test)

    print("\n  Generating visualisations...")
    plot_results(results, y_test, tfidf, df_clean)

    print("\n  NLP pipeline complete.")


if __name__ == "__main__":
    main()
