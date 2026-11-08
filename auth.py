"""Demo authentication and role authorization for Streamlit.

Set TECHACADEMY_OWNER_PASSWORD and TECHACADEMY_LEARNER_PASSWORD before deployment.
Fallback credentials are intentionally local-demo only and must be changed.
"""
import hmac
import os

DEMO_CREDENTIALS = {
    "owner": os.getenv("TECHACADEMY_OWNER_PASSWORD", "ChangeMe-Owner-V10"),
    "learner": os.getenv("TECHACADEMY_LEARNER_PASSWORD", "ChangeMe-Learner-V10"),
}


def authenticate(username, password):
    role = str(username or "").strip().lower()
    if role not in DEMO_CREDENTIALS:
        return None
    return role if hmac.compare_digest(str(password or ""), DEMO_CREDENTIALS[role]) else None


def can_access(role, page):
    owner_pages = {
        "Overview", "Gestion des apprenants", "Inscriptions & paiements",
        "Training Business Intelligence & ML", "Training Financial Intelligence",
        "Enrollment Risk & Fraud", "Learner & Operations Insights"
    }
    learner_pages = {"Conseiller IA", "Formations recommandées", "Catalogue & inscriptions"}
    return page in (owner_pages if role == "owner" else learner_pages if role == "learner" else set())
