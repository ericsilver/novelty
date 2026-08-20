"""Group themes into readable bundles: Internet, AI, blockchain, and so on.

The measure assigns each filing to one of T themes, and at T=500 the themes are
fine enough that no single one is "the internet theme": internet vocabulary is
spread over seven themes, the largest holding 27.9% of it. Anyone who wants to
ask what happened to internet trademarks therefore has to read seven rows and
add them up, which is the wrong unit for the question.

A bundle is that sum, defined once. Each bundle names a set of vocabulary terms
chosen by hand, and a theme joins the bundle when enough of the theme's own
probability mass sits on those terms.

Two things about this construct are worth stating plainly, because they bound
what a bundle page can support.

It is curated, not estimated. The term lists below are judgement calls, and a
different list gives a different bundle. They are written out in full here so
the judgement is inspectable rather than buried in a threshold. Lexical traps
are the reason for the care: a regex for carbon collects carbon fiber and
carbonated beverages, a regex for token collects coin-operated machines, and a
regex for wallet collects leather goods. Those are excluded by hand.

It is a dominant-theme count, so it undercounts. A filing enters a bundle when
its single highest-weight theme belongs to that bundle. A filing that is 30%
internet and 40% retail services is counted as retail. Each bundle reports the
share of its own vocabulary that sits in themes above the threshold, so the
reader can see how much was left out rather than having to assume it away.

Usage:  python scripts/theme_bundles.py [T]
Output: paper/results/theme_bundles_T{T}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

# A theme joins a bundle when this share of its mass sits on bundle terms.
# Set where the ranked list breaks: above it the themes read as belonging to the
# bundle, below it they are general-purpose themes carrying one bundle term.
MIN_MASS = 0.015

BUNDLES = {
    "internet": {
        "label": "Internet and the web",
        "note": "Online commerce and communication, from the 1995 web onward.",
        "terms": """internet, internet access, internet based, internet advertising,
        internet applications, internet broadcasting, internet business,
        internet chat, internet chatrooms, internet communication,
        internet communications, internet content, internet data,
        internet databases, internet domain, internet forums, internet games,
        internet marketing, internet media, internet portal, internet portals,
        internet protocol, internet radio, internet search, internet security,
        internet server, internet service, internet services, internet site,
        internet sites, internet software, internet telephony, internet users,
        internet web, online, online access, online advertising, online auction,
        online auctions, online banking, online bulletin, online business,
        online chat, online commerce, online communication, online communities,
        online community, online computer, online dating, online delivery,
        online directories, online directory, online discussion, online education,
        online electronic, online forums, online games, online gaming,
        online information, online journals, online magazine, online marketing,
        online marketplace, online newsletters, online ordering, online payment,
        online publication, online publications, online retail, online retailer,
        online sales, online service, online services, online shopping,
        online store, online stores, online trading, online training,
        online video, web, web based, web browser, web browsers, web content,
        web design, web development, web hosting, web page, web pages,
        web portal, web portals, web server, web servers, web services,
        web site, web sites, website, websites, world wide, world wide web,
        www, e-commerce, e-commerce platform, e-commerce platforms,
        e-commerce services, e-commerce software, e-commerce transactions,
        ecommerce, electronic commerce, browser, browsers, domain names,
        internet domain names, portal, portals, on-line""",
    },
    "ai": {
        "label": "Artificial intelligence",
        "note": "Explicit AI vocabulary. Thin before roughly 2015.",
        "terms": """artificial intelligence, machine learning, deep learning,
        neural, neural networks, natural language, computer vision,
        predictive analytics, chatbot, chatbot software, generative,
        algorithm, algorithms, algorithms software, expert system,
        expert systems, speech recognition, voice recognition,
        pattern recognition, image recognition, data mining""",
    },
    "blockchain": {
        "label": "Blockchain and cryptocurrency",
        "note": "Distributed-ledger vocabulary. Effectively begins 2014.",
        # Excluded by hand: token, tokens, wallet, cryptographic. The first two
        # are coin-operated machines and leather goods; the last is generic
        # security vocabulary that predates the bundle by decades.
        "terms": """blockchain, block chain, blockchains, blockchain technology,
        blockchain-based, blockchain-based smart, blockchain-based software,
        cryptocurrency, cryptocurrencies, cryptocurrency exchange,
        cryptocurrency mining, cryptocurrency payment, cryptocurrency trading,
        cryptocurrency transactions, cryptocurrency wallet, crypto,
        crypto assets, crypto collectibles, crypto tokens, crypto-collectibles,
        digital currency, distributed ledger, non-fungible, non-fungible token,
        non-fungible tokens, smart contracts, tokens blockchain,
        virtual currency, bitcoin, cryptographic keys, cryptography,
        cryptography software""",
    },
    "mobile": {
        "label": "Mobile and wireless",
        "note": "Handsets and the applications written for them.",
        "terms": """mobile application, mobile applications, mobile apps,
        mobile device, mobile devices, mobile phone, mobile phones,
        mobile telephone, mobile telephones, mobile computing, mobile computer,
        smartphone, smartphones, smartphone software, smartphone camera,
        smartphone tablet, smartphones mobile, cellular, cellular phone,
        cellular phones, cellular telephone, cellular telephones,
        cellular communication, cellular communications,
        cellular telecommunications, cellular wireless, handheld,
        handheld computer, handheld computers, handheld devices,
        handheld digital, handheld electronic, handheld mobile,
        handheld personal, handheld wireless, wireless, wireless communication,
        wireless communications, wireless devices, wireless internet,
        wireless network, wireless networks, wireless telephone,
        wireless transmission, tablet computer, tablet computers""",
    },
    "cloud": {
        "label": "Cloud and hosted software",
        "note": "Software delivered as a service rather than shipped.",
        "terms": """cloud computing, cloud computer, saas,
        software as a service, hosted, hosted computer, server hosting,
        virtualization, data center, data centers, application service,
        remote hosting, web hosting, hosting services""",
    },
    "biotech": {
        "label": "Biotechnology",
        "note": "Molecular and genomic vocabulary; a pre-internet comparison.",
        "terms": """biotechnology, biotechnological, biotechnological research,
        biotechnology companies, biotechnology fields, biotechnology industry,
        biotechnology research, genomic, genomics, gene therapy, molecular,
        molecular biology, recombinant, stem cell, stem cells, clinical trial,
        clinical trials, pharmaceutical research, diagnostic reagents,
        monoclonal, antibodies, dna, rna, proteins""",
    },
    "green": {
        "label": "Clean energy and climate",
        "note": "Renewables and emissions. Excludes carbon fiber and carbonated drinks.",
        "terms": """solar, solar cells, solar energy, solar panel, solar panels,
        solar power, photovoltaic, photovoltaic cells, photovoltaic modules,
        renewable, renewable energy, wind turbine, wind turbines, wind energy,
        wind power, biofuel, biofuels, biodiesel, ethanol fuel,
        carbon capture, carbon credits, carbon dioxide, carbon emissions,
        carbon footprint, carbon offsets, carbon offsetting,
        carbon sequestration, greenhouse gas, greenhouse gases,
        emissions reduction, electric vehicle, electric vehicles,
        electric vehicle charging, sustainable, sustainability, recycling,
        recycled, recyclable, energy efficiency, energy efficient,
        geothermal, fuel cell, fuel cells""",
    },
}


def parse_terms(s: str) -> list[str]:
    return [t.strip() for t in " ".join(s.split()).split(",") if t.strip()]


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    mp = PROC / ("topic_model.joblib" if T == 50 else f"topic_model_T{T}.joblib")
    m = joblib.load(mp)
    V, comp = m["vocabulary"], m["lda"].components_
    inv = {v: k for k, v in V.items()}
    # components_ holds unnormalised word weights per theme, so a share needs
    # the row total rather than the raw entry.
    share = comp / comp.sum(axis=1, keepdims=True)

    out = {"T": T, "min_mass": MIN_MASS, "bundles": {}}
    for name, spec in BUNDLES.items():
        terms = parse_terms(spec["terms"])
        idx = [V[t] for t in terms if t in V]
        missing = [t for t in terms if t not in V]
        mass = share[:, idx].sum(axis=1)      # share of each theme on bundle terms
        # How the bundle's vocabulary is spread over themes, which is what
        # decides whether the threshold leaves much behind.
        wt = comp[:, idx].sum(axis=1)
        wt = wt / wt.sum()
        order = np.argsort(mass)[::-1]
        keep = [int(k) for k in order if mass[k] >= MIN_MASS]
        out["bundles"][name] = {
            "label": spec["label"], "note": spec["note"],
            "n_terms": len(idx), "missing_terms": missing,
            "themes": keep,
            "theme_mass": {int(k): float(mass[k]) for k in keep},
            "captured": float(wt[keep].sum()) if keep else 0.0,
            "top_excluded": [
                {"theme": int(k), "mass": float(mass[k]),
                 "share_of_bundle": float(wt[k]),
                 "words": ", ".join(inv[j] for j in np.argsort(comp[k])[::-1][:8])}
                for k in order[len(keep):len(keep) + 3]],
            "detail": [
                {"theme": int(k), "mass": float(mass[k]),
                 "share_of_bundle": float(wt[k]),
                 "words": ", ".join(inv[j] for j in np.argsort(comp[k])[::-1][:10])}
                for k in keep],
        }
        b = out["bundles"][name]
        print("")
        print(f"### {spec['label']}  ({len(idx)} terms matched, {len(missing)} absent)")
        print(f"    {len(keep)} themes at mass >= {MIN_MASS:g}; "
              f"they carry {b['captured']:.1%} of the bundle's vocabulary")
        for d in b["detail"][:8]:
            print(f"      t{d['theme']:<4} mass {d['mass']:.3f}  "
                  f"{d['share_of_bundle']:.1%} of bundle | {d['words']}")
        if b["top_excluded"]:
            e = b["top_excluded"][0]
            print(f"      -- first excluded: t{e['theme']} mass {e['mass']:.3f} "
                  f"| {e['words']}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"theme_bundles_T{T}.json").write_text(json.dumps(out, indent=1))
    print("")
    print(f"[done] theme_bundles_T{T}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
