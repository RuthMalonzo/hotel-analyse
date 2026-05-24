import pandas as pd
import json
from django.shortcuts import render
from django.conf import settings
from textblob import TextBlob


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
def load_data():
    df = pd.read_excel(settings.DATA_FILE)
    df["note"] = pd.to_numeric(df["note"], errors="coerce")
    df = df.dropna(subset=["note", "commentaire"])
    df["commentaire"] = df["commentaire"].astype(str)
    return df


def get_sentiment(texte):
    try:
        blob = TextBlob(texte)
        pol = round(blob.sentiment.polarity, 3)
        sub = round(blob.sentiment.subjectivity, 3)
        if pol > 0.1:
            label = "Positif"
        elif pol < -0.1:
            label = "Négatif"
        else:
            label = "Neutre"
        return pol, sub, label
    except:
        return 0.0, 0.0, "Neutre"


# ─────────────────────────────────────────────
# ACCUEIL
# ─────────────────────────────────────────────
def accueil(request):
    df = load_data()
    hotels = df.groupby("hotel").agg(
        nb_avis=("commentaire", "count"),
        note_moy=("note", "mean")
    ).round(2).reset_index()
    hotels = hotels.rename(columns={
        "hotel": "hotel",
        "nb_avis": "nb_avis",
        "note_moy": "note_moy"
    })
    context = {
        "total_avis": len(df),
        "nb_hotels": df["hotel"].nunique(),
        "note_moyenne": round(df["note"].mean(), 2),
        "hotels": hotels.to_dict("records"),
    }
    return render(request, "analyse/accueil.html", context)


# ─────────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────────
def statistiques(request):
    df = load_data()
    hotel_choisi = request.GET.get("hotel", "Tous")
    hotels_list = ["Tous"] + sorted(df["hotel"].unique().tolist())

    if hotel_choisi != "Tous":
        df_f = df[df["hotel"] == hotel_choisi]
    else:
        df_f = df

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
        "stats": stats,
        "hotels_list": hotels_list,
        "hotel_choisi": hotel_choisi,
        "repartition": repartition_data,
        "apercu": df_f[["hotel", "note", "commentaire"]].head(20).to_dict("records"),
    }
    return render(request, "analyse/statistiques.html", context)


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def visualisations(request):
    df = load_data()

    # Notes moyennes par hôtel
    moy_hotel = df.groupby("hotel")["note"].mean().round(2).reset_index()
    moy_hotel_data = moy_hotel.to_dict("records")

    # Nombre d'avis par hôtel
    nb_hotel = df.groupby("hotel")["note"].count().reset_index()
    nb_hotel.columns = ["hotel", "nb_avis"]
    nb_hotel_data = nb_hotel.to_dict("records")

    # Distribution des notes
    dist = df["note"].value_counts().sort_index()
    dist_data = [{"note": int(k), "count": int(v)} for k, v in dist.items()]

    context = {
        "moy_hotel":  json.dumps(moy_hotel_data),
        "nb_hotel":   json.dumps(nb_hotel_data),
        "dist_notes": json.dumps(dist_data),
    }
    return render(request, "analyse/visualisations.html", context)


# ─────────────────────────────────────────────
# SENTIMENTS
# ─────────────────────────────────────────────
def sentiments(request):
    df = load_data()
    df["polarite"], df["subjectivite"], df["sentiment"] = zip(
        *df["commentaire"].apply(lambda x: get_sentiment(x))
    )

    positifs = (df["sentiment"] == "Positif").sum()
    neutres  = (df["sentiment"] == "Neutre").sum()
    negatifs = (df["sentiment"] == "Négatif").sum()

    # Sentiments par hôtel
    sent_hotel = df.groupby(["hotel", "sentiment"]).size().reset_index(name="count")
    sent_hotel_data = sent_hotel.to_dict("records")

    # Top avis positifs et négatifs
    top_positifs = df.nlargest(5, "polarite")[["hotel", "note", "polarite", "commentaire"]].to_dict("records")
    top_negatifs = df.nsmallest(5, "polarite")[["hotel", "note", "polarite", "commentaire"]].to_dict("records")

    context = {
        "positifs":       int(positifs),
        "neutres":        int(neutres),
        "negatifs":       int(negatifs),
        "total":          len(df),
        "pct_positifs":   round(positifs / len(df) * 100, 1),
        "pct_neutres":    round(neutres  / len(df) * 100, 1),
        "pct_negatifs":   round(negatifs / len(df) * 100, 1),
        "sent_hotel":     json.dumps(sent_hotel_data),
        "top_positifs":   top_positifs,
        "top_negatifs":   top_negatifs,
        "pol_moyenne":    round(df["polarite"].mean(), 3),
    }
    return render(request, "analyse/sentiments.html", context)


# ─────────────────────────────────────────────
# INTERACTIVE
# ─────────────────────────────────────────────
def interactive(request):
    df = load_data()
    pol_moyenne = round(df["commentaire"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    ).mean(), 3)

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