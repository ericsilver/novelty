import sys, json
from pathlib import Path
import joblib, numpy as np
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(r"C:\shared-secure\research\tm-vocabulary")
PROC = REPO / "data" / "processed"
sys.path.insert(0, str(REPO / "src"))

text = ("computerized on line search and ordering service featuring the wholesale and retail "
        "distribution of books, music, motion pictures, multimedia products and computer software "
        "in the form of printed books, audiocassettes, videocassettes, compact disks, floppy disks, "
        "CD ROMs, and direct digital transmission")

m = joblib.load(PROC / "topic_model.joblib")
lda = m["lda"]
comp = lda.components_

def report(name, vec):
    X = vec.transform([text])
    theta = lda.transform(X)[0]
    feats = vec.get_feature_names_out()
    order = np.argsort(theta)[::-1]
    print(f"--- {name} ---")
    for k in order[:10]:
        tw = [feats[i] for i in comp[k].argsort()[-6:][::-1]]
        flag = " >1/50" if theta[k] > 0.02 else ""
        print(f"topic {k:2d}  {theta[k]:.4f}{flag}  {tw}")
    n_above = int((theta > 0.02).sum())
    print(f"topics above 0.02: {n_above}")

# construction A: combination_measures.py style
vecA = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                       token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b", ngram_range=(1, 2))
report("combination_measures-style (token_pattern)", vecA)

# construction B: exhibit_encodings.py style (production for the exhibit)
from novelty.dictionary import STOPWORDS, _make_analyzer
analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
vecB = CountVectorizer(analyzer=analyzer, vocabulary=m["vocabulary"])
report("exhibit_encodings-style (analyzer)", vecB)
