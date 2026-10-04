import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st


# Configuration de la page
st.set_page_config(
    page_title="Prédiction de souscription à un dépôt à terme",
    page_icon="🏦",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine learning (XGBoost) permet de prédire si un client "
    "souscrira à un dépôt à terme, à partir de ses informations personnelles "
    "et des détails de la campagne marketing téléphonique."
)


# Chargement des artefacts (mis en cache : chargés une seule fois)

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")   # encodeurs (variables catégorielles + y)
    scaler = jb.load("scaler.joblib")       # normaliseur
    xgb_model = jb.load("xgb_model.joblib") # modèle
    return encoders, scaler, xgb_model


encoders, scaler, xgb_model = load_artifacts()
target_class = encoders['y'].classes_  # noms des classes cibles (no / yes)

colonnes_modele = [
    'age', 'job', 'marital', 'education', 'housing', 'loan', 'contact',
    'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]


# Fonction de prédiction simple

def pred_func(age, job, marital, education, housing, loan, contact,
              month, day_of_week, duration, campaign, pdays, previous, poutcome):

    job_encoded = encoders['job'].transform([job])[0]
    marital_encoded = encoders['marital'].transform([marital])[0]
    education_encoded = encoders['education'].transform([education])[0]
    housing_encoded = encoders['housing'].transform([housing])[0]
    loan_encoded = encoders['loan'].transform([loan])[0]
    contact_encoded = encoders['contact'].transform([contact])[0]
    month_encoded = encoders['month'].transform([month])[0]
    day_of_week_encoded = encoders['day_of_week'].transform([day_of_week])[0]
    poutcome_encoded = encoders['poutcome'].transform([poutcome])[0]

    df_complet = pd.DataFrame([[
        age, job_encoded, marital_encoded, education_encoded, housing_encoded,
        loan_encoded, contact_encoded, month_encoded, day_of_week_encoded,
        duration, campaign, pdays, previous, poutcome_encoded
    ]], columns=colonnes_modele)

    # Scaler sur les 14 colonnes (comme à l'entraînement : scaler.fit_transform(X))
    X_input = pd.DataFrame(scaler.transform(df_complet), columns=colonnes_modele)

    prediction = xgb_model.predict(X_input.values)[0]
    classe_predite = target_class[prediction]

    return classe_predite


# Fonction de prédiction multiple

def pred_func_csv(file):
    df = pd.read_csv(file)
    prediction = []

    for i, row in enumerate(df.iloc[:, :].values):
        try:
            y_pred = pred_func(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                row[7], row[8], row[9], row[10], row[11], row[12], row[13]
            )
            prediction.append(y_pred)
        except Exception as e:
            print(f"Erreur ligne {i} : {row} → {e}")
            prediction.append(None)

    df['y_predit'] = prediction
    return df


# Interface

st.title("🏦 Prédiction de souscription à un dépôt à terme")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire la souscription avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", value=30, step=1)
            job = st.selectbox("Emploi", options=list(encoders['job'].classes_))
            marital = st.selectbox("Statut marital", options=list(encoders['marital'].classes_))
            education = st.selectbox("Éducation", options=list(encoders['education'].classes_))
            housing = st.selectbox("Prêt immobilier", options=list(encoders['housing'].classes_))
            loan = st.selectbox("Prêt personnel", options=list(encoders['loan'].classes_))
            contact = st.selectbox("Type de contact", options=list(encoders['contact'].classes_))
        with col2:
            month = st.selectbox("Mois du dernier contact", options=list(encoders['month'].classes_))
            day_of_week = st.selectbox("Jour de la semaine", options=list(encoders['day_of_week'].classes_))
            duration = st.number_input("Durée du dernier appel (secondes)", value=0, step=1)
            campaign = st.number_input("Nombre de contacts (campagne)", value=1, step=1)
            pdays = st.number_input("Jours depuis le dernier contact", value=999, step=1)
            previous = st.number_input("Nombre de contacts précédents", value=0, step=1)
            poutcome = st.selectbox("Résultat de la précédente campagne", options=list(encoders['poutcome'].classes_))

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = pred_func(age, job, marital, education, housing, loan, contact,
                                  month, day_of_week, duration, campaign, pdays, previous, poutcome)
            st.success(f"**Souscription prédite :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire la souscription avec plusieurs entrées")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "age, job, marital, education, housing, loan, contact, month, "
        "day_of_week, duration, campaign, pdays, previous, poutcome."
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")