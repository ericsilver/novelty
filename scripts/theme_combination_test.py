"""Does the score read combinations of themes, or only which themes are present?

A natural hope for a theme-based measure is that it catches a filing built from
ordinary parts in an unusual arrangement -- the Amazon case, where every word is
common and the invention is in the assembly. Whether it does is testable.

For each filing take its top two themes and compute how often that pair occurs
together relative to what the two themes' separate frequencies would predict:

    lift = P(a and b together) / (P(a) * P(b))

A filing pairing two themes that rarely go together has a low lift. If the score
sees combination, lift should carry information about atypicality beyond how
rare the two themes are individually -- and unusual pairings, the low-lift ones,
should be the atypical ones.

Raw lift does correlate with atypicality at +0.40, but that is confounded: lift
is mechanically high when both themes are rare, and rare themes raise
atypicality on their own. The test is therefore the increment over the two
marginal rarities.

Output: printed; the figures are quoted in the construct section.
"""
import polars as pl, numpy as np, joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

P = Path('C:/shared-secure/research/tm-vocabulary/data/processed')
CLS, N = '009', 150_000
s = pl.read_parquet(P / f'rolling_surprise_class{CLS}.parquet',
                    columns=['serial_number', 'topic_kl_vs_past', 'topic_kl_vs_future']
                    ).filter(pl.col('topic_kl_vs_past').is_finite()
                             & pl.col('topic_kl_vs_future').is_finite()).sample(N, seed=42)
d = pl.read_parquet(P / f'tm_class{CLS}.parquet',
                    columns=['serial_number', 'goods_services']).join(
    s, on='serial_number', how='inner').filter(pl.col('goods_services').is_not_null())
m = joblib.load(P / 'topic_model.joblib')
vec = CountVectorizer(vocabulary=m['vocabulary'], lowercase=True,
                      token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b", ngram_range=(1, 2))
th = m['lda'].transform(vec.transform(d['goods_services'].to_list()))
th = np.clip(th, 1e-12, None); th /= th.sum(axis=1, keepdims=True)

present = th >= 0.10
marg = np.maximum(present.mean(axis=0), 1e-9)
co = (present.astype(np.float32).T @ present.astype(np.float32)) / len(th)
o = np.argsort(th, axis=1); t1, t2 = o[:, -1], o[:, -2]
lift = np.maximum(co[t1, t2], 1e-9) / (marg[t1] * marg[t2])
ll = np.log(lift)
r1, r2 = -np.log(marg[t1]), -np.log(marg[t2])      # rarity of each theme
A = 0.5 * (d['topic_kl_vs_past'].to_numpy() + d['topic_kl_vs_future'].to_numpy())
ok = np.isfinite(ll) & np.isfinite(A)
A, ll, r1, r2 = A[ok], ll[ok], r1[ok], r2[ok]

def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    r = y - X @ b
    n, k = X.shape
    V = np.linalg.pinv(X.T @ X) * (r @ r) / (n - k)
    return b, np.sqrt(np.diag(V)), 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))

z = lambda v: (v - v.mean()) / v.std()
zA, zl, zr1, zr2 = z(A), z(ll), z(r1), z(r2)
b, se, r2_ = ols([zl], zA)
print(f"n = {len(A):,}")
print(f"  lift alone            beta {b[1]:+.4f} (t {b[1]/se[1]:+6.1f})   R2 {r2_:.4f}")
b, se, r2_ = ols([zr1, zr2], zA)
print(f"  rarity of both themes                                  R2 {r2_:.4f}")
b, se, r2_ = ols([zl, zr1, zr2], zA)
print(f"  lift + rarity         beta {b[1]:+.4f} (t {b[1]/se[1]:+6.1f})   R2 {r2_:.4f}")
print(f"                        rarity1 {b[2]:+.4f}  rarity2 {b[3]:+.4f}")
