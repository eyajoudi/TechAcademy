import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta, datetime
from auth import authenticate, can_access
from services.database import connect, fetch_interactions, fetch_events, fetch_payment_requests, get_payment_request
from services.demo_data import seed_demo_data
from services.voice import transcribe_audio, synthesize_speech
from orchestrator import TrainingCenterOrchestrator, PRODUCTS
from ml_forecasting import forecast_from_records, synthetic_demo_forecast

st.set_page_config(page_title="TechAcademy AI Squad", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
theme_choice = st.sidebar.selectbox("THÈME", ["Clair", "Sombre"], index=0)


st.markdown("""
<style>
:root { color-scheme: light; }
html, body, [class*="css"] {font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:#f4f7fb !important; color:#172b43 !important;
}
.block-container {padding-top:1.35rem;padding-bottom:2rem;max-width:1500px;}
h1,h2,h3,h4,p,span,label,small,li,[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p {color:#172b43 !important; opacity:1;}
[data-testid="stCaptionContainer"] p {color:#64748b !important; opacity:1;}
[data-testid="stAlert"] p {color:inherit !important;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;border-color:#dbe4ef;}
.hero {background:linear-gradient(115deg,#0c1c33 0%,#173d63 70%,#24527a 100%);
  padding:25px 28px;border-radius:18px;color:#fff;margin-bottom:18px;
  box-shadow:0 8px 24px rgba(14,35,61,.12);}
.hero h1,.hero h1 *, .hero p {color:#fff !important; opacity:1 !important;}
.hero p {color:#d7e4f3 !important;margin:5px 0 0 0;}
.kpi {background:#fff;border:1px solid #e4eaf2;border-radius:15px;padding:15px 17px;
  box-shadow:0 3px 12px rgba(17,35,58,.035);min-height:112px;}
.kpi-label {font-size:.82rem;color:#64748b !important;font-weight:600;}
.kpi-value {font-family:Inter,system-ui,sans-serif;font-size:1.75rem;color:#132842 !important;font-weight:700;margin-top:7px;}
.kpi-note {font-size:.75rem;color:#8492a6 !important;margin-top:3px;}
[data-testid="stMetric"] {background:white;border:1px solid #e4eaf2;padding:12px 14px;border-radius:13px;}
[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#172b43 !important;}
[data-testid="stSidebar"] {background:#0d2038 !important;}
[data-testid="stSidebar"] * {color:#eaf2fb !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:4px 0;}
.stApp input,.stApp textarea,.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"] > div,.stApp [data-baseweb="textarea"] > div {
  background:#fff !important;color:#172b43 !important;border-color:#cbd5e1 !important;
}
.stApp input::placeholder,.stApp textarea::placeholder {color:#8492a6 !important;opacity:1 !important;}
.stApp [data-baseweb="select"] *,.stApp [data-baseweb="popover"] * {color:#172b43 !important;}
.stApp [data-testid="stRadio"] label,.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stFileUploader"] label {color:#334155 !important;}
.stApp [data-testid="stAudioInput"] {background:#fff !important;border:1px solid #cbd5e1;border-radius:10px;padding:8px;}
div.stButton > button,div.stFormSubmitButton > button {
  border-radius:10px;font-weight:700;background:#173d63;color:#fff;border:1px solid #173d63;
}
div.stButton > button:hover,div.stFormSubmitButton > button:hover {background:#24527a;color:#fff;border-color:#24527a;}
.stTabs [data-baseweb="tab"] {font-weight:600;color:#334155 !important;}
</style>
""", unsafe_allow_html=True)

plot_template = "plotly_dark" if theme_choice == "Sombre" else "plotly_white"
pio.templates.default = plot_template
px.defaults.template = plot_template
chart_background = "#142238" if theme_choice == "Sombre" else "white"

if theme_choice == "Sombre":
    st.markdown("""
    <style>
    :root { color-scheme: dark; }
    .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {background:#0b1422 !important;color:#e5edf7 !important;}
    h1,h2,h3,h4,p,span,label,small,li,[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"] p {color:#e5edf7 !important;}
    [data-testid="stCaptionContainer"] p {color:#9fb0c5 !important;}
    [data-testid="stMetric"],.kpi,[data-testid="stVerticalBlockBorderWrapper"] {background:#142238 !important;border-color:#263a55 !important;color:#e5edf7 !important;}
    .kpi-value,[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#f4f8ff !important;}
    .stApp input,.stApp textarea,.stApp [data-baseweb="select"] > div,.stApp [data-baseweb="input"] > div,.stApp [data-baseweb="textarea"] > div {background:#142238 !important;color:#e5edf7 !important;border-color:#40546d !important;}
    .stApp [data-baseweb="select"] *,.stApp [data-baseweb="popover"] * {color:#e5edf7 !important;}
    [data-testid="stDataFrame"] {filter:brightness(.9);}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_application():
    database = connect()
    seed_demo_data(database)
    return database, TrainingCenterOrchestrator(database)

@st.cache_data(ttl=4, show_spinner=False)
def load_dashboard_data(_database):
    return (
        fetch_interactions(_database),
        fetch_events(_database, limit=300),
        fetch_payment_requests(_database)
    )

@st.cache_data(ttl=3600, show_spinner=False)
def cached_transcription(audio_bytes):
    return transcribe_audio(audio_bytes)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_speech(text):
    return synthesize_speech(text)

conn, orchestrator = initialize_application()
records, events, payment_requests = load_dashboard_data(conn)
df = pd.DataFrame(records) if records else pd.DataFrame()
ev = pd.DataFrame(events) if events else pd.DataFrame()

def money(v):
    return f"{float(v):,.0f} TND"

def show_kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def status_label(s):
    return {
        "succeeded":"Réussi", "failed":"Refusé", "pending":"En attente",
        "pending_review":"En revue", "pending_security":"Contrôle sécurité",
        "pending_human_approval":"Validation humaine", "awaiting_owner_approval":"Validation propriétaire",
        "blocked_by_security":"Bloqué par sécurité", "not_started":"Non démarré", "approved_pending_customer_payment":"Approuvé — paiement client requis", "rejected_by_owner":"Refusé par Owner"
    }.get(str(s), str(s))

if "authenticated_role" not in st.session_state:
    st.session_state["authenticated_role"] = None

if not st.session_state["authenticated_role"]:
    st.title("TechAcademy AI Squad — Connexion")
    st.caption("Authentification de démonstration avec séparation des rôles Owner et Apprenant.")
    with st.form("login_form"):
        username = st.selectbox("Profil", ["learner", "owner"], format_func=lambda x: "Apprenant" if x == "learner" else "Owner")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", use_container_width=True)
    if submitted:
        role = authenticate(username, password)
        if role:
            st.session_state["authenticated_role"] = role
            st.rerun()
        else:
            st.error("Identifiants incorrects.")
    st.info("Comptes de démonstration locaux : owner / ChangeMe-Owner-V10 et learner / ChangeMe-Learner-V10. Changez ces mots de passe avec les variables d'environnement avant tout déploiement.")
    st.stop()

role = st.session_state["authenticated_role"]
with st.sidebar:
    st.markdown("## ◈ TECHACADEMY AI SQUAD V13")
    st.caption("AUTONOMOUS TRAINING OPERATIONS")
    st.write("Connecté : **" + ("Owner" if role == "owner" else "Apprenant") + "**")
    if st.button("Se déconnecter"):
        st.session_state["authenticated_role"] = None
        st.rerun()
    workspace = "Owner Console" if role == "owner" else "Espace apprenant"
    st.divider()
    if workspace == "Owner Console":
        st.caption("OWNER CONSOLE · Pilotage du centre")
        page = st.radio(
            "NAVIGATION",
            ["Overview", "Gestion des apprenants", "Inscriptions & paiements",
             "Training Business Intelligence & ML", "Training Financial Intelligence", "Enrollment Risk & Fraud", "Learner & Operations Insights"],
            key="owner_navigation"
        )
    else:
        st.caption("CLIENT PORTAL · Espace prospect/client")
        page = st.radio(
            "ESPACE APPRENANT",
            ["Conseiller IA", "Formations recommandées", "Catalogue & inscriptions"],
            key="client_navigation"
        )
    st.divider()
    st.markdown("**RÔLES DE LA DÉMO**")
    st.caption("Owner : pilotage des inscriptions, revenus, risques, indicateurs et analyses.")
    st.caption("Apprenant : découverte des formations, conseil et suivi des demandes d’inscription.")
    st.caption("Prototype partagé : authentification et permissions réelles non configurées.")
    st.divider()
    st.caption("Données fictives uniquement • Aucun paiement réel")

st.markdown(
    '<div class="hero"><h1>◈ TechAcademy AI Squad</h1>'
    '<p>AI-powered training guidance · Secure enrollment · Training business intelligence</p></div>',
    unsafe_allow_html=True
)

st.caption("DEMONSTRATION ENVIRONMENT · Les données préchargées sont fictives et les paiements sont simulés.")
if not can_access(role, page):
    st.error("Accès non autorisé pour ce rôle.")
    st.stop()


if not df.empty:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    total = len(df)
    interested = int((df["interest_level"] == "Intéressé").sum())
    succeeded_df = df[df["payment_status"] == "succeeded"]
    revenue = float(succeeded_df["amount"].sum())
    successful = len(succeeded_df)
    conversion = successful / total * 100 if total else 0
    failed = int((df["payment_status"] == "failed").sum())
    pending = int(df["payment_status"].isin([
        "pending", "pending_review", "pending_security",
        "pending_human_approval", "awaiting_owner_approval"
    ]).sum())
else:
    total=interested=successful=failed=pending=0
    revenue=conversion=0.0


if page == "Overview":
    st.subheader("Executive Overview")
    st.caption("Vue synthétique des performances commerciales et de l'activité du système.")
    c = st.columns(6)
    with c[0]: show_kpi("Interactions", f"{total}", "Parcours apprenants")
    with c[1]: show_kpi("Clients intéressés", f"{interested}", "Intérêt déclaré")
    with c[2]: show_kpi("Inscriptions réglées", f"{successful}", "Démonstration")
    with c[3]: show_kpi("Revenus simulés", money(revenue), "Inscriptions réglées")
    with c[4]: show_kpi("Conversion", f"{conversion:.1f}%", "Paiements / interactions")
    security_mask = ev["agent_name"].astype(str).str.contains("Security", case=False, na=False) if not ev.empty else pd.Series(dtype=bool)
    security_events = ev[security_mask] if not ev.empty else pd.DataFrame()
    security_alerts = int(security_events["status"].astype(str).str.contains(r"block|review|error", case=False, regex=True).sum()) if not security_events.empty else 0
    with c[5]: show_kpi("Alertes sécurité", f"{security_alerts}", "Décisions block / review / error")
    if security_alerts:
        st.warning(f"⚠️ Notification Owner : {security_alerts} événement(s) de sécurité nécessitent votre attention. Consultez Inscriptions & paiements et le journal de sécurité.")

    st.write("")
    left, right, sentiment_panel = st.columns([1.35,1,1])
    with left:
        st.markdown("### Revenue & activity")
        if not df.empty:
            daily = df.copy()
            daily["day"] = daily["created_at"].dt.date
            daily_rev = daily[daily["payment_status"]=="succeeded"].groupby("day",as_index=False)["amount"].sum()
            if not daily_rev.empty:
                fig = px.area(daily_rev, x="day", y="amount", markers=True,
                              labels={"day":"Date","amount":"Revenue (TND)"})
                fig.update_traces(line_color="#e99a22", fillcolor="rgba(233,154,34,.15)")
                fig.update_layout(height=320,margin=dict(l=10,r=10,t=15,b=10),paper_bgcolor=chart_background,plot_bgcolor=chart_background)
                st.plotly_chart(fig,use_container_width=True)
            else:
                st.info("Aucun revenu réussi à afficher pour le moment.")
        else:
            st.info("Le graphique apparaîtra après les premiers parcours clients.")
    with right:
        st.markdown("### Enrollment payment distribution")
        if not df.empty:
            counts = df["payment_status"].map(status_label).value_counts().reset_index()
            counts.columns=["Statut","Nombre"]
            fig=px.pie(counts,names="Statut",values="Nombre",hole=.62)
            fig.update_layout(height=320,margin=dict(l=5,r=5,t=15,b=5),showlegend=True)
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Aucune transaction enregistrée.")
    with sentiment_panel:
        st.markdown("### Customer sentiment")
        if not df.empty:
            sent = df["sentiment"].fillna("Non analysé").value_counts().rename_axis("Sentiment").reset_index(name="Interactions")
            sent_fig = px.pie(sent, names="Sentiment", values="Interactions", hole=.62,
                              color_discrete_sequence=["#16a085", "#e99a22", "#d95d5d", "#94a3b8"])
            sent_fig.update_layout(height=320, margin=dict(l=5,r=5,t=15,b=5), showlegend=True)
            st.plotly_chart(sent_fig, use_container_width=True)
        else:
            st.info("Les sentiments apparaîtront après les interactions.")

    l,r=st.columns(2)
    with l:
        st.markdown("### Main customer objections")
        if not df.empty:
            obs=[]
            for arr in df["objections"]:
                obs.extend([x for x in arr if x!="Aucune objection explicite détectée"])
            if obs:
                odf=pd.Series(obs).value_counts().rename_axis("Objection").reset_index(name="Occurrences")
                st.plotly_chart(px.bar(odf,x="Occurrences",y="Objection",orientation="h",
                                       color_discrete_sequence=["#244f78"]),use_container_width=True)
            else: st.success("Aucune objection explicite enregistrée.")
        else: st.info("Pas encore de données.")
    with r:
        st.markdown("### Latest customer journeys")
        if not df.empty:
            cols=["created_at","customer_name","product_name","interest_level","payment_status"]
            st.dataframe(df[cols].head(6),use_container_width=True,hide_index=True)
        else: st.info("Les derniers parcours apparaîtront ici.")


    st.markdown("### Recommandations pour le centre")
    strategy_items = []
    if not df.empty:
        for _, record in df.head(8).iterrows():
            for recommendation in record.get("recommendations", []) or []:
                if recommendation not in strategy_items:
                    strategy_items.append(recommendation)
    if strategy_items:
        for recommendation in strategy_items[:5]:
            st.markdown(f"- {recommendation}")
    else:
        st.info("Les recommandations stratégiques apparaîtront après les premiers parcours clients.")

    st.markdown("### Security monitoring")
    if not security_events.empty:
        security_view = security_events[["created_at", "run_id", "event_type", "status", "payload_json"]].head(10)
        st.dataframe(security_view, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun événement de sécurité enregistré. Le module externe Security & Fraud Detection doit être connecté pour produire ses décisions et alertes.")

elif page == "Gestion des apprenants":
    st.subheader("Gestion des apprenants")
    st.caption("Recherche, segmentation et consultation des parcours clients.")
    if df.empty:
        st.info("Aucun client enregistré. Créez un parcours depuis Conseiller IA.")
    else:
        search=st.text_input("Rechercher un client ou un service")
        f1,f2,f3=st.columns(3)
        with f1: interest_filter=st.multiselect("Niveau d'intérêt",sorted(df["interest_level"].dropna().unique()))
        with f2: channel_filter=st.multiselect("Canal",sorted(df["channel"].dropna().unique()))
        with f3: sentiment_filter=st.multiselect("Sentiment",sorted(df["sentiment"].dropna().unique()))
        filtered=df.copy()
        if search:
            filtered=filtered[
                filtered["customer_name"].astype(str).str.contains(search,case=False,na=False) |
                filtered["product_name"].astype(str).str.contains(search,case=False,na=False)
            ]
        if interest_filter: filtered=filtered[filtered["interest_level"].isin(interest_filter)]
        if channel_filter: filtered=filtered[filtered["channel"].isin(channel_filter)]
        if sentiment_filter: filtered=filtered[filtered["sentiment"].isin(sentiment_filter)]
        st.caption(f"{len(filtered)} parcours correspondant aux filtres.")
        st.dataframe(filtered[["created_at","customer_name","product_name","channel","interest_level","sentiment","payment_status"]],use_container_width=True,hide_index=True)
        names=filtered["customer_name"].dropna().astype(str).unique().tolist()
        if names:
            selected=st.selectbox("Consulter le profil d'un client",names)
            profile=filtered[filtered["customer_name"].astype(str)==selected]
            latest=profile.iloc[0]
            a,b,c=st.columns(3)
            a.metric("Interactions",len(profile))
            b.metric("Intérêt le plus récent",latest["interest_level"])
            c.metric("Canal récent",latest["channel"])
            st.markdown("#### Dernier parcours")
            st.write("Service :",latest["product_name"])
            st.write("Sentiment :",latest["sentiment"])
            analysis = latest.get("customer_analysis", {}) or {}
            if analysis:
                st.markdown("#### Analyse avancée du sentiment et des intentions")
                aa, bb, cc = st.columns(3)
                aa.metric("Score sentiment", f"{analysis.get('sentiment_score', 0)}/100")
                bb.metric("Score intention", f"{analysis.get('intent_score', 0)}/100")
                cc.metric("Urgence", analysis.get("urgency", "Normale"))
                st.write("**Intention détectée :**", analysis.get("intent_label", "Non déterminée"))
                st.write("**Émotions :**", ", ".join(analysis.get("emotions", [])))
                st.write("**Indices :**", " · ".join(analysis.get("evidence", [])))
                st.write("**Prochaine action recommandée :**", analysis.get("next_best_action", "—"))
            st.write("Objections :",", ".join(latest["objections"]))
            st.write("Motivations :",", ".join(latest["motivations"]))
            st.write("Message client :",latest.get("customer_message") or "Non disponible dans les anciennes données.")
            st.write("Réponse commerciale :",latest.get("sales_reply") or "Non disponible dans les anciennes données.")
            st.write("Recommandations :",latest["recommendations"])


elif page == "Inscriptions & paiements":
    st.subheader("Inscriptions & paiements")
    st.caption("File de validation Owner : chaque décision est enregistrée. Les paiements restent simulés.")
    pending = [r for r in payment_requests if r["status"] in ("awaiting_owner_approval", "pending_security")]
    if pending:
        st.markdown("### Demandes en attente de décision")
        for request in pending:
            with st.container(border=True):
                left, right = st.columns([3, 2])
                left.markdown(f"**{request['product_name']} — Demande #{request['id']}**")
                left.caption(f"Sécurité : {request['security_decision']} · Risque : {request['risk_level']} · Statut : {status_label(request['status'])}")
                left.info("Évaluation Owner : " + str(request.get("owner_assessment", "manual_review")) + " — " + str(request.get("owner_assessment_reason", "À examiner")))
                if request.get("security_reasons"):
                    left.caption("Motifs : " + " · ".join(request["security_reasons"]))
                reason = right.text_area("Motif / note de décision", key=f"owner_reason_{request['id']}",
                                         placeholder="Expliquez brièvement votre décision.", height=75)
                if request["status"] == "pending_security":
                    right.warning("Revue sécurité requise avant la validation Owner.")
                    if right.button("Autoriser après revue sécurité", key=f"security_review_{request['id']}"):
                        try:
                            orchestrator.review_security_request(request["id"], "Owner", reason)
                            load_dashboard_data.clear()
                            st.success("Contrôle sécurité autorisé après revue humaine. La demande passe à la validation Owner.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                    if right.button("Refuser la demande", key=f"owner_reject_security_{request['id']}"):
                        try:
                            result = orchestrator.reject_payment_request(request["id"], "Owner", reason)
                            load_dashboard_data.clear()
                            st.warning(result["message"])
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                elif request["status"] == "awaiting_owner_approval":
                    right.caption("Contrôle sécurité favorable. Décidez si la demande peut être acceptée.")
                    c1, c2 = right.columns(2)
                    if c1.button("Approuver", key=f"owner_approve_{request['id']}", type="primary"):
                        try:
                            result = orchestrator.approve_payment_request(request["id"], approved_by="Owner")
                            load_dashboard_data.clear()
                            st.success(result["message"])
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                    if c2.button("Refuser", key=f"owner_reject_{request['id']}"):
                        try:
                            result = orchestrator.reject_payment_request(request["id"], "Owner", reason)
                            load_dashboard_data.clear()
                            st.warning(result["message"])
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
    else:
        st.success("Aucune demande en attente de décision.")

    st.markdown("### Approuvées — en attente du paiement client")
    approved = [r for r in payment_requests if r["status"] == "approved_pending_customer_payment"]
    if approved:
        for request in approved:
            st.info(f"Demande #{request['id']} · {request['product_name']} — L'apprenant doit confirmer le paiement simulé depuis son espace.")
    else:
        st.caption("Aucune demande approuvée en attente de paiement.")

    st.divider()
    st.markdown("### Historique des décisions et paiements")
    history = [r for r in payment_requests if r["status"] not in ("awaiting_owner_approval", "pending_security")]
    if history:
        hist_rows = [{"Demande":r["id"],"Formation":r["product_name"],"Sécurité":r["security_decision"],
                      "Risque":r["risk_level"],"Statut":status_label(r["status"]),
                      "Décideur":r.get("human_approved_by") or "—",
                      "Motif":r.get("owner_assessment_reason") or "—"} for r in history]
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Les décisions apparaîtront ici.")
    st.markdown("### Journal des événements récents")
    if not ev.empty:
        event_view = ev[["created_at", "run_id", "agent_name", "event_type", "status", "duration_ms"]].head(20)
        st.dataframe(event_view, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun événement enregistré.")

elif page == "Training Business Intelligence & ML":
    st.subheader("Training Business Intelligence & Machine Learning")
    st.caption("Analyse des ventes, des objections et démonstration de prévision de revenus.")
    if not df.empty:
        l,r=st.columns(2)
        with l:
            st.markdown("#### Interactions par offre")
            by_product=df.groupby("product_name",as_index=False).size().rename(columns={"size":"interactions"})
            st.plotly_chart(px.bar(by_product,x="product_name",y="interactions",color_discrete_sequence=["#e99a22"]),use_container_width=True)
        with r:
            st.markdown("#### Intérêt client")
            interest_counts=df["interest_level"].value_counts().rename_axis("Niveau").reset_index(name="Clients")
            st.plotly_chart(px.pie(interest_counts,names="Niveau",values="Clients",hole=.5),use_container_width=True)
    st.divider()
    st.markdown("### Revenue Forecasting — Machine Learning")
    st.caption("Le modèle de démonstration est entraîné sur une série synthétique. La prévision issue des données réelles n'est affichée que lorsque l'historique est suffisant.")
    demo_forecast, demo_history = synthetic_demo_forecast(periods=7)
    fig_demo = go.Figure()
    fig_demo.add_trace(go.Scatter(x=demo_history["day"], y=demo_history["revenue"],
                                  mode="lines+markers", name="Historique synthétique"))
    fig_demo.add_trace(go.Scatter(x=list(range(15,22)), y=demo_forecast["predicted_revenue"],
                                  mode="lines+markers", name="Prévision ML synthétique"))
    fig_demo.update_layout(xaxis_title="Jour", yaxis_title="Revenu fictif (TND)", height=360,
                           margin=dict(l=10,r=10,t=25,b=10))
    st.plotly_chart(fig_demo, use_container_width=True)
    st.info("Démonstration ML : données synthétiques, distinctes des revenus réels. Ce graphique prouve le fonctionnement du pipeline, pas une prévision financière réelle.")
    recorded_forecast, recorded_error = forecast_from_records(records, periods=7)
    if recorded_error:
        st.warning("Prévision sur l'historique enregistré indisponible : " + recorded_error)
    else:
        history_frame = recorded_forecast["history"].copy()
        history_frame["type"] = "Historique enregistré"
        future_frame = recorded_forecast["forecast"].copy()
        future_frame["type"] = "Prévision ML"
        recorded_chart = pd.concat([
            history_frame.rename(columns={"date":"day","amount":"revenue"})[["day","revenue","type"]],
            future_frame.rename(columns={"date":"day","predicted_revenue":"revenue"})[["day","revenue","type"]]
        ])
        fig_recorded = px.line(
            recorded_chart, x="day", y="revenue", color="type", markers=True,
            color_discrete_map={"Historique enregistré":"#244f78","Prévision ML":"#e99a22"}
        )
        fig_recorded.update_layout(xaxis_title="Date", yaxis_title="Revenu (TND)", height=360)
        st.plotly_chart(fig_recorded, use_container_width=True)
        st.success(
            f"Prévision générée avec {recorded_forecast['model']} sur "
            f"{recorded_forecast['training_days']} journées enregistrées."
        )
        st.caption("Les données préchargées de démonstration sont fictives. Les résultats ne constituent pas une prévision financière fiable sans historique réel plus long.")

elif page == "Training Financial Intelligence":
    st.subheader("Training Financial Intelligence")
    st.caption("Analyse de la performance financière, de la qualité des paiements et de la contribution des offres.")
    from agents.financial_intelligence import FinancialIntelligenceAgent
    fi = FinancialIntelligenceAgent().run(records)
    a,b,c,d = st.columns(4)
    a.metric("Revenu simulé", money(fi["simulated_revenue"]))
    b.metric("Ticket moyen réussi", money(fi["average_successful_ticket"]))
    c.metric("Taux de succès", f"{fi['payment_success_rate_percent']:.1f}%")
    d.metric("Taux d'échec", f"{fi['failure_rate_percent']:.1f}%")
    st.markdown("### Performance par offre")
    if fi["revenue_by_offer"]:
        offer_frame=pd.DataFrame(fi["revenue_by_offer"])
        left,right=st.columns(2)
        with left: st.plotly_chart(px.bar(offer_frame,x="offer",y="revenue",title="Revenu simulé par offre"),use_container_width=True)
        with right: st.plotly_chart(px.bar(offer_frame,x="offer",y="conversion_percent",title="Conversion par offre"),use_container_width=True)
        st.dataframe(offer_frame,use_container_width=True,hide_index=True)
    st.markdown("### Insights et décisions suggérées")
    for insight in fi["insights"]: st.info(insight)
    st.caption(fi["disclaimer"])


elif page == "Enrollment Risk & Fraud":
    st.subheader("Enrollment Risk & Fraud")
    st.caption("Scoring automatique déclenché lors d'une demande d'inscription avec paiement. Aucun paramètre de risque n'est saisi manuellement.")

    fraud_events = pd.DataFrame()
    if not ev.empty:
        fraud_events = ev[
            (ev["agent_name"].astype(str).str.contains("ML Fraud Detection Agent", case=False, na=False)) &
            (ev["event_type"].astype(str) == "risk_scoring_completed")
        ].copy()

    if fraud_events.empty:
        st.info("Aucune évaluation disponible pour le moment. Le scoring sera lancé automatiquement lorsqu'un apprenant confirmera une demande de paiement dans le parcours d'inscription.")
        st.markdown("**Données utilisées automatiquement :** prix de la formation, heure serveur, volume des demandes récentes et tentatives de paiement échouées enregistrées.")
    else:
        import json
        parsed = []
        for _, row in fraud_events.iterrows():
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except (TypeError, ValueError):
                payload = {}
            parsed.append({"created_at": row.get("created_at"), "run_id": row.get("run_id"),
                           "status": row.get("status"), "payload": payload})

        latest = parsed[0]["payload"]
        x, y, z = st.columns(3)
        x.metric("Score de risque", f"{float(latest.get('risk_score', 0)):.1f}/100")
        y.metric("Niveau", str(latest.get("risk_level", "inconnu")).upper())
        z.metric("Signal ML", str(latest.get("decision", "—")).upper())
        st.caption(f"Dernière évaluation : {parsed[0]['created_at']} · Source : données automatiques de la plateforme")

        st.markdown("#### Facteurs explicatifs")
        for signal in latest.get("signals", []):
            st.write(f"- {signal}")

        st.markdown("#### Variables analysées automatiquement")
        features = latest.get("features", {})
        availability = latest.get("feature_availability", {})
        labels = {
            "amount": "Montant de la formation (TND)", "hour": "Heure serveur",
            "transactions_10m": "Demandes d'inscription / paiement (10 min)",
            "failed_attempts_24h": "Paiements échoués (24 h)",
            "new_device": "Nouvel appareil", "geo_distance_km": "Distance géographique (km)",
            "merchant_mismatch": "Incohérence marchand / opération"
        }
        feature_rows = []
        for key, value in features.items():
            feature_rows.append({"Indicateur": labels.get(key, key), "Valeur utilisée": value,
                                 "Origine": availability.get(key, "calcul automatique")})
        if feature_rows:
            st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Historique des évaluations")
        history = []
        for item in parsed[:30]:
            payload = item["payload"]
            history.append({"Date": item["created_at"], "Niveau": payload.get("risk_level", "—"),
                            "Score /100": payload.get("risk_score", "—"),
                            "Décision ML": payload.get("decision", "—"), "Run": item["run_id"]})
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

        importance_df = pd.DataFrame(latest.get("model_feature_importance", []))
        if not importance_df.empty:
            st.markdown("#### Importance globale des variables dans le modèle")
            st.plotly_chart(px.bar(importance_df, x="importance", y="feature", orientation="h",
                title="Importance des variables — Random Forest"), use_container_width=True)
            st.caption("L'importance globale décrit l'utilisation des variables par le modèle ; elle n'explique pas à elle seule une décision individuelle.")
        st.caption(latest.get("disclaimer", "Score expérimental basé sur des données synthétiques."))

    st.warning("Limite de la démonstration : les informations d'appareil et de géolocalisation ne sont pas collectées. Elles sont signalées comme indisponibles et ne sont pas présentées comme des observations réelles. Le score ML ne remplace ni le contrôle de sécurité ni la validation humaine.")


elif page == "Learner & Operations Insights":
    st.subheader("Learner & Operations Insights")
    st.caption("Vue Owner : compréhension des besoins apprenants, objections et performance des formations.")
    if df.empty:
        st.info("Aucune interaction disponible.")
    else:
        col1, col2, col3 = st.columns(3)
        sentiment_counts = df["sentiment"].fillna("Non analysé").value_counts()
        col1.metric("Sentiments détectés", int(df["sentiment"].notna().sum()))
        col2.metric("Clients à relancer", int(df["interest_level"].isin(["Hésitant", "Pas intéressé"]).sum()))
        col3.metric("Offres distinctes", int(df["product_name"].nunique()))

        left, right = st.columns(2)
        with left:
            st.markdown("#### Répartition des sentiments")
            sent_df = sentiment_counts.rename_axis("Sentiment").reset_index(name="Interactions")
            st.plotly_chart(px.pie(sent_df, names="Sentiment", values="Interactions", hole=.55),
                            use_container_width=True)
        with right:
            st.markdown("#### Performance par offre")
            offer_df = df.groupby("product_name", as_index=False).agg(
                interactions=("id", "count"),
                interested=("interest_level", lambda x: int((x == "Intéressé").sum())),
                successful=("payment_status", lambda x: int((x == "succeeded").sum()))
            )
            st.plotly_chart(px.bar(offer_df, x="product_name", y=["interactions", "interested", "successful"],
                                   barmode="group"), use_container_width=True)

        st.markdown("#### Évolution du sentiment client")
        sentiment_rows = []
        for record in records:
            analysis = record.get("customer_analysis", {}) or {}
            sentiment_rows.append({
                "Date": pd.to_datetime(record.get("created_at"), errors="coerce"),
                "Score indicatif": analysis.get("sentiment_score"),
                "Sentiment": record.get("sentiment", "Non analysé")
            })
        sentiment_df = pd.DataFrame(sentiment_rows).dropna(subset=["Date"])
        if not sentiment_df.empty:
            daily_sentiment = sentiment_df.groupby(sentiment_df["Date"].dt.date, as_index=False)["Score indicatif"].mean()
            daily_sentiment.columns = ["Date", "Score moyen indicatif"]
            st.plotly_chart(px.line(daily_sentiment, x="Date", y="Score moyen indicatif",
                                    markers=True, range_y=[0,100]),
                            use_container_width=True)
        st.caption("Les scores proviennent du moteur lexical et restent indicatifs ; ils ne constituent pas une mesure psychologique.")

        st.markdown("#### Intelligence client avancée — NLP hybride")
        st.caption("Le moteur combine l'analyse lexicale explicable et un classifieur TF-IDF + Logistic Regression entraîné sur un petit corpus synthétique.")
        ai_rows=[]
        for record in records[:30]:
            analysis=record.get("customer_analysis",{}) or {}
            advanced=analysis.get("advanced_ai",{}) or {}
            if advanced:
                ai_rows.append({"Date":record.get("created_at"),"Client":record.get("customer_name"),
                    "Classe NLP":advanced.get("label"),"Confiance indicative (%)":advanced.get("confidence"),
                    "Thèmes":", ".join(advanced.get("topics",[]))})
        if ai_rows: st.dataframe(pd.DataFrame(ai_rows),use_container_width=True,hide_index=True)
        else: st.info("Les analyses NLP enrichies apparaîtront sur les nouvelles interactions.")

        st.markdown("#### Objections principales")
        objections = []
        for values in df["objections"]:
            objections.extend([v for v in values if not str(v).startswith("Aucune")])
        if objections:
            obj = pd.Series(objections).value_counts().rename_axis("Objection").reset_index(name="Volume")
            st.plotly_chart(px.bar(obj, x="Volume", y="Objection", orientation="h"), use_container_width=True)
        else:
            st.info("Aucune objection récurrente identifiée.")

        st.markdown("#### Interactions nécessitant un suivi")
        followup = df[df["interest_level"].isin(["Hésitant", "Pas intéressé"])].copy()
        if not followup.empty:
            st.dataframe(followup[["created_at", "customer_name", "product_name", "sentiment",
                                   "interest_level", "channel", "payment_status"]],
                         use_container_width=True, hide_index=True)
            st.download_button("Exporter la liste de suivi",
                               followup.to_csv(index=False).encode("utf-8"),
                               "customer_followup.csv", "text/csv")
        else:
            st.success("Aucun suivi prioritaire selon les critères simples de la démonstration.")


elif page == "Formations recommandées":
    st.subheader("Recommandation personnalisée")
    st.caption("Décrivez vos objectifs : l’agent compare votre profil aux formations informatiques disponibles et explique ses propositions.")
    with st.form("client_recommendation_form"):
        goal = st.text_area(
            "Quel est votre objectif de formation ou le domaine informatique qui vous intéresse ?",
            placeholder="Ex. : Je connais les bases de Python et je souhaite me spécialiser en intelligence artificielle ou en cybersécurité.",
            height=120
        )
        budget = st.number_input("Budget maximal (TND) — facultatif, 0 si non défini",
                                 min_value=0, max_value=10000, value=0, step=50)
        recommendation_submit = st.form_submit_button("Analyser mon besoin", use_container_width=True)
    if recommendation_submit:
        from agents.offer_recommendation import OfferRecommendationAgent
        from agents.customer_intelligence import CustomerIntelligenceAgent
        intelligence = CustomerIntelligenceAgent().run(goal, "À déterminer")
        result = OfferRecommendationAgent().recommend(
            goal, PRODUCTS, intelligence, budget=budget if budget > 0 else None, limit=3
        )
        st.session_state["client_recommendation_result"] = result
        st.session_state["client_recommendation_goal"] = goal
    result = st.session_state.get("client_recommendation_result")
    if result:
        st.markdown("### Formations recommandées")
        for item in result["recommendations"]:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{item['rank']}. {item['name']}**")
                c1.write(item["description"])
                c1.caption(f"{item.get('duration', 'Durée à confirmer')} · {item.get('format', 'Format à confirmer')} · {item['price']:.0f} {item['currency']}")
                c1.caption("Critères correspondants : " + " · ".join(item["reasons"]))
                c2.metric("Correspondance", f"{item['score']}/100")
                c2.caption(item["confidence_label"])
                if st.button("Consulter cette solution", key=f"client_reco_{item['product_key']}", use_container_width=True):
                    st.session_state["selected_product_key"] = item["product_key"]
                    st.success("Solution sélectionnée. Ouvrez « Catalogue & inscriptions » pour consulter les détails et créer une demande de démonstration.")
        if not result["recommendations"]:
            st.info("Aucune formation n'atteint le seuil minimal de pertinence. Je préfère vous poser une question plutôt que de proposer une formation au hasard.")
        for question in result.get("clarifying_questions", []):
            st.write("• " + question)
        st.caption(result["disclaimer"])


elif page == "Catalogue & inscriptions":
    st.subheader("Catalogue des formations informatiques")
    st.caption("Découvrez les parcours, leurs objectifs et leurs tarifs, puis créez une demande d’inscription de démonstration.")
    product_options = {item["name"]: key for key, item in PRODUCTS.items()}
    default_key = st.session_state.get("selected_product_key", list(PRODUCTS.keys())[0])
    selected_name = st.selectbox(
        "Formation à consulter",
        list(product_options.keys()),
        index=list(product_options.values()).index(default_key)
    )
    product_key = product_options[selected_name]
    product = PRODUCTS[product_key]
    with st.container(border=True):
        st.markdown(f"### {product['name']}")
        st.markdown(f"## {product['price']:.0f} {product['currency']}")
        st.write(product["description"])
        st.caption(f"{product['duration']} · {product['format']} · Public : {product['audience']}")
        st.markdown("**Fonctionnalités principales**")
        for outcome in product.get("outcomes", []):
            st.markdown(f"- {outcome}")
    st.markdown("### Demande d’inscription et de règlement")
    with st.form("payment_request_form"):
        customer_name = st.text_input("Nom du client (fictif pour cette démonstration)", "Amira")
        st.markdown("**Contexte de risque (données fictives de démonstration)**")
        risk_a, risk_b = st.columns(2)
        risk_velocity = risk_a.number_input("Autres demandes en 10 minutes", min_value=0, max_value=20, value=1)
        risk_failed = risk_b.number_input("Tentatives échouées en 24 h", min_value=0, max_value=20, value=0)
        risk_c, risk_d = st.columns(2)
        risk_new_device = risk_c.checkbox("Nouvel appareil")
        risk_distance = risk_d.number_input("Distance inhabituelle (km)", min_value=0.0, max_value=2000.0, value=0.0, step=10.0)
        risk_mismatch = st.checkbox("Incohérence entre demandeur et inscription")
        confirmed = st.checkbox(
            f"Je confirme vouloir créer une demande d’inscription de démonstration de {product['price']:.0f} {product['currency']}."
        )
        st.caption("Aucune carte bancaire, aucun code secret et aucune donnée bancaire réelle ne sont demandés.")
        payment_submit = st.form_submit_button("Soumettre au contrôle de sécurité", use_container_width=True)
    if payment_submit:
        if not confirmed:
            st.error("Veuillez confirmer explicitement la demande avant de continuer.")
        elif not customer_name.strip():
            st.error("Veuillez saisir un nom de démonstration.")
        else:
            ctx = orchestrator.run(
                customer_name, product_key,
                "Je souhaite créer une demande d’inscription de démonstration à cette formation informatique.",
                "Intéressé", "text", True,
                risk_features={"hour": datetime.now().hour, "transactions_10m": risk_velocity,
                    "failed_attempts_24h": risk_failed, "new_device": risk_new_device,
                    "geo_distance_km": risk_distance, "merchant_mismatch": risk_mismatch}
            )
            st.session_state["last_payment_request"] = {
                "request_id": ctx.payment_request_id,
                "security": ctx.security_result,
                "payment": ctx.payment_result,
                "product": product["name"]
            }
            load_dashboard_data.clear()
            st.rerun()
    st.divider()
    st.markdown("### Demande d’inscription personnalisée")
    st.caption("Vous pouvez aussi demander une inscription personnalisée ou un entretien d’orientation, indépendamment du catalogue.")
    with st.form("custom_training_request"):
        op_customer = st.text_input("Référence client fictive", "Client Démo")
        op_type = st.selectbox(
            "Type de demande",
            ["Inscription à une formation", "Demande d’orientation", "Demande de formation en entreprise", "Demande de report simulé"]
        )
        op_amount = st.number_input("Montant estimatif (TND)", min_value=1.0, max_value=100000.0, value=150.0, step=10.0)
        op_note = st.text_area("Votre objectif et votre contexte", "Le prospect souhaite s’inscrire à une formation et vérifier les modalités de son inscription.")
        op_confirm = st.checkbox("Je confirme la création de cette demande de démonstration.")
        op_submit = st.form_submit_button("Soumettre au contrôle de sécurité", use_container_width=True)
    if op_submit:
        if not op_confirm or not op_customer.strip():
            st.error("Confirmez la demande et renseignez une référence client fictive.")
        else:
            operation_product = {
                "id": "custom_" + op_type.lower().replace(" ", "_"),
                "name": op_type,
                "price": float(op_amount),
                "currency": "TND",
                "description": op_note,
                "duration": "Opération ponctuelle",
                "format": "Simulation locale",
                "audience": "Client de démonstration",
                "outcomes": ["Contrôle de sécurité", "Validation humaine", "Suivi du statut"]
            }
            ctx = orchestrator.run(
                op_customer, "ai_fundamentals", op_note, "Intéressé", "text", True,
                product_override=operation_product,
                risk_features={"hour": datetime.now().hour, "transactions_10m": 1,
                    "failed_attempts_24h": 0, "new_device": False,
                    "geo_distance_km": 0, "merchant_mismatch": False}
            )
            st.session_state["last_payment_request"] = {
                "request_id": ctx.payment_request_id,
                "security": ctx.security_result,
                "payment": ctx.payment_result,
                "product": op_type
            }
            load_dashboard_data.clear()
            st.rerun()

    request_info = st.session_state.get("last_payment_request")
    if request_info:
        st.markdown("### Suivi de votre demande")
        st.write(f"Formation : **{request_info['product']}**")
        st.write(f"Référence de demande : **{request_info['request_id']}**")
        decision = request_info["security"].get("decision", "review")
        if decision == "allow":
            st.success("Contrôle sécurité autorisé. La demande attend encore la validation humaine du propriétaire.")
        elif decision == "block":
            st.error("La demande a été bloquée par le contrôle de sécurité.")
        else:
            st.warning("La demande est en attente du module Security & Fraud Detection de l'équipe.")
        st.caption("Le règlement simulé ne sera exécuté qu’après autorisation sécurité et validation humaine. Aucun paiement réel n’est effectué.")

    st.divider()
    st.markdown("### Suivre une demande et confirmer le paiement")
    st.caption("Saisissez la référence communiquée après votre inscription. Aucun numéro de carte ou code bancaire n'est demandé.")
    request_id_input = st.number_input("Référence de demande", min_value=1, step=1, value=1, key="learner_request_lookup")
    if st.button("Consulter ma demande", key="lookup_learner_payment"):
        st.session_state["learner_payment_lookup"] = int(request_id_input)
    lookup_id = st.session_state.get("learner_payment_lookup")
    if lookup_id:
        learner_request = get_payment_request(conn, lookup_id)
        if not learner_request:
            st.error("Demande introuvable. Vérifiez la référence.")
        else:
            st.write(f"**Formation :** {learner_request['product_name']}")
            st.write(f"**Statut :** {status_label(learner_request['status'])}")
            if learner_request["status"] == "approved_pending_customer_payment":
                st.success("Votre demande est approuvée. Vous pouvez confirmer le paiement de démonstration.")
                consent = st.checkbox("Je confirme vouloir effectuer le paiement simulé, sans saisir de donnée bancaire réelle.",
                                      key=f"customer_payment_consent_{lookup_id}")
                if st.button("Confirmer le paiement simulé", key=f"customer_pay_{lookup_id}", type="primary"):
                    if not consent:
                        st.error("Veuillez confirmer votre accord.")
                    else:
                        try:
                            result = orchestrator.complete_customer_payment(lookup_id, customer_confirmed=True)
                            load_dashboard_data.clear()
                            st.success(result["message"])
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
            elif learner_request["status"] == "rejected_by_owner":
                st.error("Votre demande a été refusée. Aucun paiement n'a été effectué.")
            elif learner_request["status"] in ("succeeded", "failed", "pending"):
                st.info("Résultat du paiement simulé : " + status_label(learner_request["status"]))
            else:
                st.info("Votre demande est en cours de traitement. Le paiement sera disponible après approbation.")



elif page == "Conseiller IA":
    st.subheader("Conseiller IA — Espace client")
    st.caption("Décrivez votre projet professionnel, obtenez des informations sur les programmes et échangez avec notre conseiller IA.")

    product_options = {item["name"]: key for key, item in PRODUCTS.items()}
    selected_name = st.selectbox(
        "Formation concernée par votre question",
        list(product_options.keys()),
        index=list(product_options.values()).index(st.session_state.get("selected_product_key", list(PRODUCTS.keys())[0]))
    )
    product_key = product_options[selected_name]
    st.session_state["selected_product_key"] = product_key
    selected_product = PRODUCTS[product_key]
    st.caption(f"{selected_product['duration']} · {selected_product['format']} · {selected_product['price']:.0f} {selected_product['currency']}")

    st.markdown("### Parler à notre assistant")
    for chat_item in st.session_state.get("assistant_history", []):
        with st.chat_message(chat_item["role"]):
            st.write(chat_item["content"])
    with st.form("client_journey"):
        left, right = st.columns(2)
        with left:
            customer = st.text_input("Votre nom (nom fictif pour la démonstration)", "Amira")
            interest = st.radio("Votre niveau d'intérêt", ["Intéressé", "Hésitant", "Pas intéressé"], horizontal=True)
        with right:
            channel = st.radio("Canal de communication", ["text", "voice"],
                               format_func=lambda x: "Texte" if x == "text" else "Voix", horizontal=True)
            message = st.text_area(
                "Votre message",
                st.session_state.get("chat_draft", ""),
                height=115,
                placeholder="Ex. : Je veux apprendre le RAG avec Python. Je suis débutant et je cherche un parcours pratique."
            )
        submitted = st.form_submit_button("Envoyer mon message", use_container_width=True)

    st.markdown("#### Entrée vocale")
    voice_consent = st.checkbox(
        "Je consens à l'enregistrement temporaire de mon message vocal pour cette démonstration.",
        key="voice_recording_consent"
    )
    if voice_consent:
        audio = st.audio_input("Enregistrer un message en français")
        if audio is not None and st.button("Transcrire le message vocal"):
            try:
                transcript = cached_transcription(audio.getvalue())
                st.session_state["voice_transcript"] = transcript
                st.success("Transcription terminée.")
                st.write(transcript)
            except Exception as e:
                st.error(f"Transcription indisponible : {e}. Le mode texte reste disponible.")
    else:
        st.caption("L'enregistrement vocal reste désactivé tant que le consentement n'est pas donné.")
    if "voice_transcript" in st.session_state:
        if st.button("Utiliser cette transcription"):
            st.session_state["message_prefill"] = st.session_state["voice_transcript"]
            st.rerun()

    if submitted:
        actual = st.session_state.get("message_prefill", message)
        history = st.session_state.get("assistant_history", [])
        ctx = orchestrator.run(
            customer, product_key, actual, interest, channel, False,
            conversation_history=history
        )
        history.append({"role": "user", "content": actual})
        history.append({"role": "assistant", "content": ctx.sales_result["message"]})
        st.session_state["assistant_history"] = history[-12:]
        load_dashboard_data.clear()
        st.session_state["last_result"] = {
            "sales": ctx.sales_result, "intelligence": ctx.intelligence_result,
            "security": ctx.security_result, "payment": ctx.payment_result,
            "financial": ctx.financial_result, "strategy": ctx.strategy_result,
            "recommendations": ctx.offer_recommendation_result,
            "run_id": ctx.run_id
        }
        st.session_state.pop("message_prefill", None)
        st.success("Votre message a été traité et le parcours a été enregistré.")

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.markdown("### Réponse de l'assistant")
        st.caption("Mode assistant : " + ("LLM Groq avec repli local" if r["sales"].get("assistant_mode") == "groq_llm" else "FAQ locale fondée sur le catalogue"))
        if st.button("Écouter la réponse commerciale"):
            try:
                st.audio(cached_speech(r["sales"]["message"]), format="audio/mp3")
            except Exception as e:
                st.warning(f"Synthèse vocale indisponible : {e}")
        a, b = st.columns(2)
        with a:
            st.markdown("**Analyse du besoin client**")
            ci = r["intelligence"]
            m1, m2 = st.columns(2)
            m1.metric("Sentiment", ci.get("sentiment", "—"), f"Score indicatif : {ci.get('sentiment_score', 0)}/100")
            m2.metric("Intention", ci.get("intent_label", ci.get("intent", "—")), f"Signal : {ci.get('intent_score', 0)}/100")
            st.write("**Émotions :**", ", ".join(ci.get("emotions", [])))
            st.write("**Urgence :**", ci.get("urgency", "Normale"))
            st.write("**Objections :**", ", ".join(ci.get("objections", [])))
            st.write("**Motivations :**", ", ".join(ci.get("motivations", [])))
            st.write("**Prochaine action :**", ci.get("next_best_action", "—"))
            st.caption(f"Méthode : {ci.get('analysis_method', 'Analyse lexicale explicable')}")
            with st.expander("Détails et indices"):
                st.write("Confiance heuristique :", f"{ci.get('sentiment_confidence', 0)}%")
                for item in ci.get("evidence", []):
                    st.markdown(f"- {item}")
                st.caption("Analyse explicable par règles et mots-clés ; scores indicatifs, non probabilités calibrées.")
        with b:
            st.markdown("**Frontière externe de cybersécurité**")
            st.warning("Module réservé au collaborateur. Il n'est pas développé dans cette version.")
            st.json(r["security"])
        st.markdown("**État du paiement de démonstration**")
        st.write(r["payment"])
        st.markdown("**Recommandations stratégiques**")
        for rec in r["strategy"]["recommendations"]:
            st.success(rec)

        st.markdown("### Formations correspondant à votre besoin")
        offer_result = r.get("recommendations", {})
        offer_items = offer_result.get("recommendations", [])
        if offer_items:
            for item in offer_items:
                with st.container(border=True):
                    st.markdown(f"**{item['rank']}. {item['name']}** — {item['price']:.0f} {item['currency']}")
                    st.caption(f"Correspondance indicative : {item['score']}/100 · {item['confidence_label']}")
                    st.write(item["description"])
                    st.caption("Pourquoi cette formation ? " + " · ".join(item.get("reasons", [])))
                    st.caption(f"{item.get('duration', 'Durée à confirmer')} · {item.get('format', 'Format à confirmer')}")
        else:
            st.info("Je préfère ne pas classer des formations sans critères suffisants. " +
                    "Précisez votre domaine, votre niveau ou votre objectif pour obtenir une recommandation pertinente.")
        for question in offer_result.get("clarifying_questions", []):
            st.write("• " + question)
        st.caption(offer_result.get("disclaimer", ""))
