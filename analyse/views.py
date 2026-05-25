import pandas as pd
import json
import os
from django.shortcuts import render
from django.conf import settings
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from langdetect import detect

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
def load_data():
    df_fr = pd.read_excel(settings.DATA_FILE)
    df_en = pd.read_excel(settings.DATA_FILE_EN)
    df_fr["langue"] = "🇫🇷 Français"
    df_en["langue"] = "🇬🇧 Anglais"
    df = pd.concat([df_fr, df_en], ignore_index=True)
    df["note"] = pd.to_numeric(df["note"], errors="coerce")
    df = df.dropna(subset=["note", "commentaire"])
    df["commentaire"] = df["commentaire"].astype(str)
    return df


# ─────────────────────────────────────────────
# ANALYSE SENTIMENT
# 🇫🇷 TextBlob-fr pour français
# 🇬🇧 TextBlob pour anglais
# ─────────────────────────────────────────────
def get_sentiment(texte):
    try:
        langue = detect(str(texte))
        if langue == "fr":
            blob = TextBlob(str(texte), pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
            pol  = round(blob.sentiment[0], 3)
            sub  = round(blob.sentiment[1], 3)
        else:
            blob = TextBlob(str(texte))
            pol  = round(blob.sentiment.polarity, 3)
            sub  = round(blob.sentiment.subjectivity, 3)
        if pol > 0.05:
            label = "Positif"
        elif pol < -0.05:
            label = "Négatif"
        else:
            label = "Neutre"
        return pol, sub, label
    except:
        return 0.0, 0.0, "Neutre"


# ─────────────────────────────────────────────
# ANALYSE DES ASPECTS
# ─────────────────────────────────────────────
def analyser_aspects(df):
    aspects = {
        "service":       ["service", "personnel", "staff", "accueil", "réception", "reception"],
        "chambre":       ["chambre", "room", "lit", "bed", "espace"],
        "nourriture":    ["nourriture", "restaurant", "repas", "food", "cuisine", "breakfast"],
        "piscine":       ["piscine", "pool", "baignade"],
        "propreté":      ["propre", "propreté", "clean", "hygiène", "hygiene"],
        "prix":          ["prix", "tarif", "cher", "coût", "abordable", "price", "expensive"],
        "localisation":  ["localisation", "location", "quartier", "centre", "situé"],
        "wifi":          ["wifi", "internet", "connexion", "network"],
        "climatisation": ["climatisation", "clim", "ac", "froid", "chaud", "air conditioning"],
        "parking":       ["parking", "voiture", "stationnement", "car park"],
        "general":       ["bien", "good", "excellent", "parfait", "super", "great", "nice"],
    }
    counts = {}
    for aspect, mots in aspects.items():
        total = 0
        for commentaire in df["commentaire"]:
            commentaire_lower = str(commentaire).lower()
            for mot in mots:
                if mot in commentaire_lower:
                    total += 1
                    break
        counts[aspect] = total
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# ACCUEIL
# ─────────────────────────────────────────────
def accueil(request):
    df = load_data()
    langue_choisie = request.GET.get("langue", "Tous")
    langues_list   = ["Tous", "🇫🇷 Français", "🇬🇧 Anglais"]

    if langue_choisie != "Tous":
        df_filtre = df[df["langue"] == langue_choisie]
    else:
        df_filtre = df

    hotels = df_filtre.groupby("hotel").agg(
        nb_avis=("commentaire", "count"),
        note_moy=("note", "mean"),
        langue=("langue", lambda x: ", ".join(x.unique()))
    ).round(2).reset_index()

    context = {
        "total_avis":     len(df_filtre),
        "nb_hotels":      df_filtre["hotel"].nunique(),
        "note_moyenne":   round(df_filtre["note"].mean(), 2),
        "hotels":         hotels.to_dict("records"),
        "langues_list":   langues_list,
        "langue_choisie": langue_choisie,
    }
    return render(request, "analyse/accueil.html", context)


# ─────────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────────
def statistiques(request):
    df = load_data()
    hotel_choisi = request.GET.get("hotel", "Tous")
    hotels_list  = ["Tous"] + sorted(df["hotel"].unique().tolist())

    if hotel_choisi != "Tous":
        df_f = df[df["hotel"] == hotel_choisi].copy()
    else:
        df_f = df.copy()

    df_f["note"] = df_f["note"].round(0).astype(int)

    stats = {
        "nb_avis":    len(df_f),
        "moyenne":    round(df_f["note"].mean(), 2),
        "ecart_type": round(df_f["note"].std(), 2),
        "variance":   round(df_f["note"].var(), 2),
        "mediane":    round(df_f["note"].median(), 2),
        "minimum":    int(df_f["note"].min()),
        "maximum":    int(df_f["note"].max()),
        "q1":         round(df_f["note"].quantile(0.25), 2),
        "q3":         round(df_f["note"].quantile(0.75), 2),
        "iqr":        round(df_f["note"].quantile(0.75) - df_f["note"].quantile(0.25), 2),
        "cv":         round((df_f["note"].std() / df_f["note"].mean()) * 100, 2),
    }

    repartition = df_f["note"].value_counts().sort_index()
    repartition_data = [
        {"note": int(k), "count": int(v), "pct": round(v / len(df_f) * 100, 1)}
        for k, v in repartition.items()
    ]

    context = {
        "stats":        stats,
        "hotels_list":  hotels_list,
        "hotel_choisi": hotel_choisi,
        "repartition":  repartition_data,
        "apercu":       df_f[["hotel", "note", "commentaire"]].head(20).to_dict("records"),
    }
    return render(request, "analyse/statistiques.html", context)


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def visualisations(request):
    df = load_data()

    moy_hotel      = df.groupby("hotel")["note"].mean().round(2).reset_index()
    moy_hotel_data = moy_hotel.to_dict("records")

    nb_hotel         = df.groupby("hotel")["note"].count().reset_index()
    nb_hotel.columns = ["hotel", "nb_avis"]
    nb_hotel_data    = nb_hotel.to_dict("records")

    dist      = df["note"].value_counts().sort_index()
    dist_data = [{"note": int(k), "count": int(v)} for k, v in dist.items()]

    context = {
        "moy_hotel":  json.dumps(moy_hotel_data),
        "nb_hotel":   json.dumps(nb_hotel_data),
        "dist_notes": json.dumps(dist_data),
    }
    return render(request, "analyse/visualisations.html", context)


# ─────────────────────────────────────────────
# SENTIMENTS (avec cache)
# ─────────────────────────────────────────────
def sentiments(request):
    df         = load_data()
    cache_file = os.path.join(settings.BASE_DIR, 'sentiments_cache.csv')

    # ✅ Si le cache existe → on le lit directement
    if os.path.exists(cache_file):
        print("✅ Lecture du cache...")
        df_sent = pd.read_csv(cache_file, encoding="utf-8-sig")
    else:
        # 1ère fois → analyse tout → sauvegarde
        print("⏳ 1ère analyse en cours sur tous les commentaires...")
        resultats          = df["commentaire"].apply(lambda x: get_sentiment(x))
        df["polarite"]     = resultats.apply(lambda x: x[0])
        df["subjectivite"] = resultats.apply(lambda x: x[1])
        df["sentiment"]    = resultats.apply(lambda x: x[2])
        df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        df_sent = df
        print("✅ Cache sauvegardé !")

    positifs = (df_sent["sentiment"] == "Positif").sum()
    neutres  = (df_sent["sentiment"] == "Neutre").sum()
    negatifs = (df_sent["sentiment"] == "Négatif").sum()

    sent_hotel      = df_sent.groupby(["hotel", "sentiment"]).size().reset_index(name="count")
    sent_hotel_data = sent_hotel.to_dict("records")

    top_positifs = df_sent.nlargest(5, "polarite")[["hotel", "note", "polarite", "commentaire"]].to_dict("records")
    top_negatifs = df_sent.nsmallest(5, "polarite")[["hotel", "note", "polarite", "commentaire"]].to_dict("records")

    aspects_data = analyser_aspects(df_sent)
    aspects_json = json.dumps([{"aspect": k, "count": v} for k, v in aspects_data])

    context = {
        "positifs":     int(positifs),
        "neutres":      int(neutres),
        "negatifs":     int(negatifs),
        "total":        len(df_sent),
        "pct_positifs": round(positifs / len(df_sent) * 100, 1),
        "pct_neutres":  round(neutres  / len(df_sent) * 100, 1),
        "pct_negatifs": round(negatifs / len(df_sent) * 100, 1),
        "sent_hotel":   json.dumps(sent_hotel_data),
        "top_positifs": top_positifs,
        "top_negatifs": top_negatifs,
        "pol_moyenne":  round(df_sent["polarite"].mean(), 3),
        "aspects":      aspects_json,
    }
    return render(request, "analyse/sentiments.html", context)


# ─────────────────────────────────────────────
# INTERACTIVE
# ─────────────────────────────────────────────
def interactive(request):
    pol_moyenne = 0.15

    result = None
    if request.method == "POST":
        commentaire = request.POST.get("commentaire", "")
        if commentaire.strip():
            pol, sub, label = get_sentiment(commentaire)
            result = {
                "commentaire":  commentaire,
                "polarite":     pol,
                "subjectivite": sub,
                "label":        label,
                "pol_moyenne":  pol_moyenne,
                "plus_positif": pol > pol_moyenne,
            }

    context = {
        "result":      result,
        "pol_moyenne": pol_moyenne,
    }
    return render(request, "analyse/interactive.html", context)