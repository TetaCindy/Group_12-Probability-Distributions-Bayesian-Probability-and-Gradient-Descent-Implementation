"""Bayes' theorem on IMDb reviews -- basic Python only (csv + re from the standard library)."""
import csv
import re

WORD_RE = re.compile(r"[a-z']+")


def load_reviews(path="imdb.csv"):
    """Return a list of (set_of_words, label). Using a set means a review either
    contains the keyword or it does not (document-level presence)."""
    reviews = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row["review"].lower().replace("<br />", " ")
            reviews.append((set(WORD_RE.findall(text)), row["sentiment"]))
    return reviews


def bayes_posterior(prior, likelihood, marginal):
    """Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)."""
    return likelihood * prior / marginal


def keyword_stats(reviews, keyword, target="positive"):
    """Count-based estimates of prior, likelihood, marginal and the Bayes posterior
    for P(target | keyword). Also returns the direct posterior as a sanity check."""
    n_total = len(reviews)
    n_target = sum(1 for _, label in reviews if label == target)
    n_kw = sum(1 for words, _ in reviews if keyword in words)
    n_kw_and_target = sum(1 for words, label in reviews if keyword in words and label == target)

    prior = n_target / n_total                       # P(Positive)
    likelihood = n_kw_and_target / n_target          # P(keyword | Positive)
    marginal = n_kw / n_total                        # P(keyword)
    posterior = bayes_posterior(prior, likelihood, marginal)   # P(Positive | keyword)
    direct = n_kw_and_target / n_kw                  # counted directly, must equal posterior
    return dict(keyword=keyword, n_kw=n_kw, prior=prior, likelihood=likelihood,
                marginal=marginal, posterior=posterior, direct=direct)


def format_table(stats_list, target="Positive"):
    """Markdown table, one row per keyword."""
    lines = [f"| Keyword | Prior P({target}) | Likelihood P(kw|{target}) | Marginal P(kw) | Posterior P({target}|kw) |",
             "|---|---|---|---|---|"]
    for s in stats_list:
        lines.append(f"| {s['keyword']} | {s['prior']:.4f} | {s['likelihood']:.4f} | "
                     f"{s['marginal']:.4f} | {s['posterior']:.4f} |")
    return "\n".join(lines)

