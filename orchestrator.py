from dataclasses import dataclass, field
from typing import Dict, Any
import uuid, time
from datetime import datetime, timedelta

from agents.sales import SalesAgent
from agents.customer_intelligence import CustomerIntelligenceAgent
from agents.payment import PaymentAgent, DemoPaymentProvider
from agents.financial_analyst import FinancialDataAnalyst
from agents.business_strategy import BusinessStrategyAgent
from agents.offer_recommendation import OfferRecommendationAgent
from agents.security_adapter import ExternalSecurityAdapter
from agents.fraud_detection import FraudDetectionAgent
from agents.advanced_customer_ai import AdvancedCustomerAI
from agents.financial_intelligence import FinancialIntelligenceAgent
from agents.input_guard import screen_untrusted_text
from services.payment_vault import (store_payment_request, get_private_payment,
    update_private_payment, update_private_payment_status)
from services.database import (
    save_event, save_interaction, fetch_interactions,
    create_payment_request, fetch_payment_requests,
    update_payment_request, get_payment_request, transition_payment_request
)

PRODUCTS = {
    "ai_fundamentals": {
        "id": "ai_fundamentals", "name": "Intelligence artificielle & Machine Learning",
        "price": 450.0, "currency": "TND",
        "description": "Parcours pratique pour comprendre l'IA, préparer des données, entraîner des modèles de Machine Learning et évaluer leurs performances.",
        "duration": "8 semaines", "format": "Hybride · Cours et ateliers pratiques",
        "audience": "Débutants en informatique, étudiants et professionnels en reconversion",
        "outcomes": ["Python pour l'IA", "Préparation des données", "Classification et régression", "Évaluation des modèles"]
    },
    "cybersecurity": {
        "id": "cybersecurity", "name": "Cybersécurité & Ethical Hacking",
        "price": 520.0, "currency": "TND",
        "description": "Formation défensive sur les fondamentaux réseaux, la sécurité des systèmes, l'analyse des risques et les bonnes pratiques de protection.",
        "duration": "10 semaines", "format": "Hybride · Laboratoires encadrés",
        "audience": "Étudiants, techniciens et débutants en sécurité informatique",
        "outcomes": ["Fondamentaux réseau", "Linux et durcissement", "Analyse des vulnérabilités en laboratoire autorisé", "Réponse aux incidents"]
    },
    "cloud_computing": {
        "id": "cloud_computing", "name": "Cloud Computing & AWS/Azure Fundamentals",
        "price": 490.0, "currency": "TND",
        "description": "Découverte des architectures cloud, des services de calcul et stockage, des réseaux virtuels et des principes de sécurité cloud.",
        "duration": "8 semaines", "format": "En ligne · Travaux pratiques guidés",
        "audience": "Développeurs, administrateurs systèmes et étudiants",
        "outcomes": ["Concepts IaaS/PaaS/SaaS", "Déploiement cloud", "Réseaux virtuels", "Sécurité et coûts cloud"]
    },
    "devops": {
        "id": "devops", "name": "DevOps & CI/CD",
        "price": 480.0, "currency": "TND",
        "description": "Apprentissage des pratiques d'intégration et de livraison continues, de l'automatisation et de la conteneurisation.",
        "duration": "9 semaines", "format": "Hybride · Projet fil rouge",
        "audience": "Développeurs et administrateurs systèmes débutants/intermédiaires",
        "outcomes": ["Git et GitHub", "Linux et scripting", "Docker", "Pipelines CI/CD"]
    },
    "devsecops": {
        "id": "devsecops", "name": "DevSecOps & Secure Software Delivery",
        "price": 560.0, "currency": "TND",
        "description": "Intégration des contrôles de sécurité dans le cycle de développement logiciel et les pipelines CI/CD.",
        "duration": "8 semaines", "format": "En ligne · Ateliers pratiques",
        "audience": "Développeurs, ingénieurs DevOps et étudiants avancés",
        "outcomes": ["Secure SDLC", "Analyse statique", "Gestion des dépendances", "Sécurisation des pipelines"]
    },
    "data_science": {
        "id": "data_science", "name": "Data Science & Business Intelligence",
        "price": 460.0, "currency": "TND",
        "description": "Parcours consacré à l'analyse de données, à la visualisation et à la création de tableaux de bord décisionnels.",
        "duration": "8 semaines", "format": "Hybride · Études de cas",
        "audience": "Étudiants, analystes débutants et professionnels",
        "outcomes": ["Python et Pandas", "Nettoyage des données", "Visualisation", "Power BI et KPI"]
    },
    "programming_python": {
        "id": "programming_python", "name": "Programmation Python & Développement logiciel",
        "price": 390.0, "currency": "TND",
        "description": "Formation progressive en programmation Python, algorithmique, structures de données et développement d'applications.",
        "duration": "7 semaines", "format": "En ligne · Exercices et mini-projets",
        "audience": "Débutants en programmation et étudiants",
        "outcomes": ["Bases Python", "Algorithmique", "POO", "Développement de mini-applications"]
    }
}

@dataclass
class WorkflowContext:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_name: str = ""
    product: Dict[str, Any] = field(default_factory=dict)
    channel: str = "text"
    customer_message: str = ""
    interest_level: str = "Hésitant"
    customer_confirmed_payment: bool = False
    sales_result: Dict[str, Any] = field(default_factory=dict)
    intelligence_result: Dict[str, Any] = field(default_factory=dict)
    security_result: Dict[str, Any] = field(default_factory=dict)
    fraud_result: Dict[str, Any] = field(default_factory=dict)
    advanced_ai_result: Dict[str, Any] = field(default_factory=dict)
    financial_intelligence_result: Dict[str, Any] = field(default_factory=dict)
    payment_result: Dict[str, Any] = field(default_factory=dict)
    financial_result: Dict[str, Any] = field(default_factory=dict)
    strategy_result: Dict[str, Any] = field(default_factory=dict)
    offer_recommendation_result: Dict[str, Any] = field(default_factory=dict)
    payment_request_id: int = None

class TrainingCenterOrchestrator:
    """
    Python orchestrator coordinating learner journeys, course enrollment, payment simulation, security and analytics agents.
    """
    def __init__(self, conn, payment_provider=None, security_adapter=None):
        self.conn = conn
        self.sales = SalesAgent()
        self.intelligence = CustomerIntelligenceAgent()
        self.payment = PaymentAgent(payment_provider or DemoPaymentProvider())
        self.analyst = FinancialDataAnalyst()
        self.strategy = BusinessStrategyAgent()
        self.offer_recommender = OfferRecommendationAgent()
        self.security = security_adapter or ExternalSecurityAdapter()
        self.fraud_detector = FraudDetectionAgent()
        self.advanced_ai = AdvancedCustomerAI()
        self.financial_intelligence = FinancialIntelligenceAgent()

    def _event(self, ctx, agent, event_type, payload, started=None, status="completed"):
        duration = (time.perf_counter() - started) * 1000 if started else None
        save_event(self.conn, ctx.run_id, agent, event_type, payload, status, duration)

    def review_security_request(self, request_id, reviewer, reason):
        request = get_payment_request(self.conn, request_id)
        if not request:
            raise ValueError("Demande introuvable.")
        if request["status"] != "pending_security" or request["security_decision"] != "review":
            raise PermissionError("Cette demande ne peut pas être autorisée par la revue manuelle.")
        if request["risk_level"] == "high":
            raise PermissionError("Une demande à risque élevé ne peut pas être autorisée manuellement.")
        if not str(reason or "").strip():
            raise ValueError("Un motif de revue est obligatoire.")
        changed = transition_payment_request(
            self.conn, request_id, "pending_security", "awaiting_owner_approval",
            security_decision="allow", reviewer=reviewer,
            reason="Revue sécurité humaine : " + str(reason).strip()
        )
        if not changed:
            raise ValueError("La demande a déjà été traitée.")
        update_private_payment_status(request_id, "awaiting_owner_approval")
        save_event(self.conn, request["run_id"], "Security & Fraud Detection Agent",
                   "manual_security_review", {"decision":"allow","reviewer":reviewer,
                   "reason":str(reason).strip()}, status="human_allowed")
        return get_payment_request(self.conn, request_id)

    def reject_payment_request(self, request_id, rejected_by, reason):
        request = get_payment_request(self.conn, request_id)
        if not request:
            raise ValueError("Demande introuvable.")
        if request["status"] not in ("awaiting_owner_approval", "pending_security"):
            raise PermissionError("Cette demande n'est plus en attente de décision.")
        if not str(reason or "").strip():
            raise ValueError("Un motif de refus est obligatoire.")
        expected = request["status"]
        changed = transition_payment_request(
            self.conn, request_id, expected, "rejected_by_owner",
            reviewer=rejected_by, reason="Refus Owner : " + str(reason).strip()
        )
        if not changed:
            raise ValueError("La demande a déjà été traitée.")
        update_private_payment_status(request_id, "rejected_by_owner")
        self.conn.execute("UPDATE interactions SET payment_status=? WHERE run_id=?",
                          ("rejected_by_owner", request["run_id"]))
        self.conn.commit()
        save_event(self.conn, request["run_id"], "Owner", "payment_request_rejected",
                   {"request_id":int(request_id),"reviewer":rejected_by,
                    "reason":str(reason).strip()}, status="rejected_by_owner")
        return {"status":"rejected_by_owner","message":"Demande refusée. Aucun paiement n'a été exécuté."}

    def approve_payment_request(self, request_id, approved_by="Owner Console", demo_outcome="succeeded"):
        request = get_payment_request(self.conn, request_id)
        if not request:
            raise ValueError("Demande de paiement introuvable.")
        if request["status"] != "awaiting_owner_approval" or request["security_decision"] != "allow":
            raise PermissionError("Paiement non autorisé : le contrôle sécurité doit être favorable avant la validation Owner.")
        private_payment = get_private_payment(request_id)
        if private_payment is None:
            raise ValueError("Données privées de règlement introuvables.")
        changed = transition_payment_request(
            self.conn, request_id, "awaiting_owner_approval",
            "approved_pending_customer_payment", reviewer=approved_by,
            reason="Approuvée par Owner ; paiement client encore requis."
        )
        if not changed:
            raise ValueError("La demande a déjà été traitée.")
        update_private_payment_status(request_id, "approved_pending_customer_payment")
        self.conn.execute("UPDATE interactions SET payment_status=? WHERE run_id=?",
                          ("approved_pending_customer_payment", request["run_id"]))
        self.conn.commit()
        save_event(self.conn, request["run_id"], "Owner", "payment_request_approved",
                   {"request_id":int(request_id),"approved_by":approved_by},
                   status="approved_pending_customer_payment")
        return {"status":"approved_pending_customer_payment",
                "message":"Demande approuvée. L'apprenant peut maintenant confirmer le paiement simulé."}

    def complete_customer_payment(self, request_id, customer_confirmed=True, demo_outcome="succeeded"):
        request = get_payment_request(self.conn, request_id)
        if not request:
            raise ValueError("Demande introuvable.")
        if request["status"] != "approved_pending_customer_payment" or request["security_decision"] != "allow":
            raise PermissionError("Le paiement n'est pas disponible : la demande doit d'abord être approuvée.")
        if not customer_confirmed:
            raise ValueError("Confirmation explicite du client requise.")
        private_payment = get_private_payment(request_id)
        if private_payment is None:
            raise ValueError("Données de paiement privées introuvables.")
        provider = DemoPaymentProvider(outcome=demo_outcome)
        result = PaymentAgent(provider).run(
            amount=private_payment["amount"], currency=private_payment["currency"],
            customer_reference=private_payment["customer_reference"],
            customer_confirmed=True, approved_by_security=True, human_approved=True
        )
        update_private_payment(request_id, result)
        changed = transition_payment_request(
            self.conn, request_id, "approved_pending_customer_payment",
            result["status"], reviewer=None, reason="Paiement simulé confirmé par l'apprenant."
        )
        if not changed:
            raise ValueError("Cette demande a déjà été réglée ou modifiée.")
        self.conn.execute("UPDATE interactions SET payment_status=? WHERE run_id=?",
                          (result["status"], request["run_id"]))
        self.conn.commit()
        save_event(self.conn, request["run_id"], "Payment Agent", "customer_demo_payment_completed",
                   {"request_id":int(request_id),"status":result["status"],
                    "sandbox":True,"bank_data_collected":False},
                   status=result["status"])
        return {"status":result["status"], "message":result["message"],
                "sandbox":True, "bank_data_collected":False}

    def run(self, customer_name, product_key, customer_message, interest_level,
            channel="text", customer_confirmed_payment=False, conversation_history=None,
            product_override=None, risk_features=None):
        selected_product = product_override or PRODUCTS[product_key]
        input_guard = screen_untrusted_text(customer_message)
        ctx = WorkflowContext(
            customer_name=customer_name, product=selected_product, channel=channel,
            customer_message=customer_message, interest_level=interest_level,
            customer_confirmed_payment=customer_confirmed_payment
        )

        t = time.perf_counter()
        ctx.intelligence_result = self.intelligence.run(customer_message, interest_level)
        ctx.advanced_ai_result = self.advanced_ai.analyze(customer_message)
        ctx.intelligence_result["advanced_ai"] = ctx.advanced_ai_result
        self._event(ctx, "Advanced Customer AI Agent", "nlp_analysis_completed", ctx.advanced_ai_result)
        self._event(ctx, "Customer Intelligence Agent", "customer_profile_created", ctx.intelligence_result, t)

        t = time.perf_counter()
        ctx.sales_result = self.sales.run(
            customer_name, ctx.product, customer_message,
            customer_analysis=ctx.intelligence_result,
            conversation_history=conversation_history
        )
        self._event(ctx, "Voice Sales Agent", "offer_presented", ctx.sales_result, t)

        t = time.perf_counter()
        recommendation_context = "\\n".join(
            str(item.get("content", "")) for item in (conversation_history or [])[-8:]
            if item.get("role") == "user"
        )
        recommendation_context = (recommendation_context + "\\n" + str(customer_message or "")).strip()
        ctx.offer_recommendation_result = self.offer_recommender.recommend(
            recommendation_context, PRODUCTS, ctx.intelligence_result
        )
        self._event(ctx, "Offer Recommendation Agent", "offers_ranked",
                    ctx.offer_recommendation_result, t)

        transaction_context = {
            "run_id": ctx.run_id,
            "customer_reference": customer_name,
            "amount": ctx.product["price"],
            "currency": ctx.product["currency"],
            "product_id": ctx.product["id"],
            "customer_confirmed": customer_confirmed_payment,
            "purpose": "training_course_enrollment_demo",
            "prompt_injection_screening": input_guard
        }

        if customer_confirmed_payment:
            t = time.perf_counter()
            # Automatic feature extraction from the enrollment and platform records.
            # No manual form and no collection of device fingerprint or precise location.
            now = datetime.now()
            recent_cutoff = (now - timedelta(minutes=10)).isoformat(timespec="seconds")
            day_cutoff = (now - timedelta(hours=24)).isoformat(timespec="seconds")
            recent_enrollments = self.conn.execute(
                "SELECT COUNT(*) FROM payment_requests WHERE created_at >= ?", (recent_cutoff,)
            ).fetchone()[0]
            failed_attempts = self.conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE created_at >= ? AND LOWER(COALESCE(payment_status,'')) IN ('failed','declined')",
                (day_cutoff,)
            ).fetchone()[0]
            model_features = {
                "hour": now.hour,
                "transactions_10m": int(recent_enrollments) + 1,
                "failed_attempts_24h": int(failed_attempts),
                # These signals are not collected by this demo; use neutral model values and disclose this.
                "new_device": False,
                "geo_distance_km": 0.0,
                "merchant_mismatch": False,
            }
            ctx.fraud_result = self.fraud_detector.assess({**transaction_context, **model_features})
            ctx.fraud_result["feature_source"] = "automatic_platform_data"
            ctx.fraud_result["feature_availability"] = {
                "amount": "catalogue_course_price", "hour": "server_timestamp",
                "transactions_10m": "payment_requests_created_in_last_10_minutes",
                "failed_attempts_24h": "failed_or_declined_interactions_in_last_24_hours",
                "new_device": "not_collected_neutral_value_used",
                "geo_distance_km": "not_collected_neutral_value_used",
                "merchant_mismatch": "not_applicable_to_training_center_demo"
            }
            ctx.fraud_result["prompt_injection_screening"] = input_guard
            self._event(ctx, "ML Fraud Detection Agent", "risk_scoring_completed", ctx.fraud_result, t, ctx.fraud_result.get("risk_level", "unknown"))
            try:
                security_output = self.security.review({**transaction_context, "ml_fraud_assessment": ctx.fraud_result})
                if not isinstance(security_output, dict):
                    raise TypeError("Invalid security adapter response")
                decision = str(security_output.get("decision", "review")).lower()
                if decision not in ("allow", "review", "block"):
                    decision = "review"
                ctx.security_result = {
                    **security_output,
                    "decision": decision,
                    "risk_level": security_output.get("risk_level", "unknown"),
                    "reasons": security_output.get("reasons", [])
                }
                security_event_status = decision
                # ML is an additional risk signal: high risk downgrades allow to review, never auto-approves.
                if ctx.fraud_result.get("risk_level") == "high" and decision == "allow":
                    decision = "review"
                    ctx.security_result["decision"] = "review"
                    ctx.security_result.setdefault("reasons", []).append("ML Fraud Detection : risque élevé, revue complémentaire obligatoire.")
                    security_event_status = "review_ml_override"
                if input_guard["flagged"] and decision == "allow":
                    decision = "review"
                    ctx.security_result["decision"] = "review"
                    ctx.security_result.setdefault("reasons", []).append("Texte client signalé par le filtre anti-prompt-injection : revue obligatoire.")
                    security_event_status = "review_input_guard"
            except Exception:
                ctx.security_result = {
                    "decision": "review",
                    "risk_level": "unknown",
                    "reasons": ["Module de sécurité indisponible ou réponse invalide. Demande conservée en attente."],
                    "requires_human_approval": True,
                    "adapter_status": "error"
                }
                decision = "review"
                security_event_status = "error"
            self._event(ctx, "Security & Fraud Detection Agent (external)", "security_review",
                        ctx.security_result, t, security_event_status)

            security_approved = decision == "allow"
            payment_preview = self.payment.run(
                amount=ctx.product["price"], currency=ctx.product["currency"],
                customer_reference=customer_name, customer_confirmed=True,
                approved_by_security=security_approved, human_approved=False
            )
            if decision == "block":
                request_status = "blocked_by_security"
            elif decision == "allow":
                request_status = "awaiting_owner_approval"
            else:
                request_status = "pending_security"

            # Automated owner pre-assessment: recommends a next step but does not execute/approve payment.
            ml_level = ctx.fraud_result.get("risk_level", "unknown")
            if decision == "block" or ml_level == "high":
                owner_assessment, owner_reason = "reject_or_escalate", "Risque élevé ou décision de blocage sécurité."
            elif decision == "review" or ml_level == "medium":
                owner_assessment, owner_reason = "manual_review", "Vérification complémentaire requise."
            else:
                owner_assessment, owner_reason = "eligible_for_owner_confirmation", "Contrôles automatiques favorables ; confirmation Owner toujours requise."
            ctx.payment_request_id = create_payment_request(self.conn, {
                "run_id": ctx.run_id, "product_id": ctx.product["id"],
                "product_name": ctx.product["name"], "security_decision": decision,
                "risk_level": ml_level, "security_reasons": ctx.security_result.get("reasons", []),
                "status": request_status, "owner_assessment": owner_assessment,
                "owner_assessment_reason": owner_reason
            })
            # Financial details are stored in a separate private ledger, not the Owner database.
            store_payment_request(ctx.payment_request_id, customer_name, ctx.product["price"], ctx.product["currency"], request_status)
            ctx.payment_result = {
                **payment_preview,
                "status": request_status,
                "request_id": ctx.payment_request_id,
                "security_decision": decision,
                "message": (
                    "Demande transmise au propriétaire pour validation humaine."
                    if request_status == "awaiting_owner_approval" else
                    "Demande bloquée par le contrôle de sécurité."
                    if request_status == "blocked_by_security" else
                    "Demande en attente du module Security & Fraud Detection."
                )
            }
            self._event(ctx, "Payment Agent", "payment_request_created", ctx.payment_result,
                        status=request_status)
        else:
            ctx.security_result = {
                "decision": "not_requested",
                "risk_level": "not_assessed",
                "reasons": ["Aucune demande de paiement initiée par le client."],
                "requires_human_approval": True
            }
            ctx.payment_result = self.payment.run(
                amount=ctx.product["price"], currency=ctx.product["currency"],
                customer_reference=customer_name, customer_confirmed=False
            )
            self._event(ctx, "Payment Agent", "payment_not_started", ctx.payment_result,
                        status="not_started")

        prior = fetch_interactions(self.conn)
        current = {
            "interest_level": interest_level,
            "payment_status": ctx.payment_result.get("status"),
            "amount": ctx.product["price"],
            "objections": ctx.intelligence_result.get("objections", [])
        }
        t = time.perf_counter()
        ctx.financial_result = self.analyst.run(prior + [current])
        ctx.financial_intelligence_result = self.financial_intelligence.run(prior + [current])
        self._event(ctx, "Financial Data Analyst", "kpis_calculated", ctx.financial_result, t)
        self._event(ctx, "Financial Intelligence Agent", "financial_insights_generated", ctx.financial_intelligence_result)

        t = time.perf_counter()
        ctx.strategy_result = self.strategy.run(ctx.intelligence_result, ctx.financial_result, ctx.financial_intelligence_result)
        self._event(ctx, "Business Strategy Agent", "recommendations_generated", ctx.strategy_result, t)

        save_interaction(self.conn, {
            "customer_name": customer_name, "product_name": ctx.product["name"],
            "amount": ctx.product["price"], "currency": ctx.product["currency"],
            "channel": channel, "interest_level": interest_level,
            "sentiment": ctx.intelligence_result.get("sentiment"),
            "objections": ctx.intelligence_result.get("objections", []),
            "motivations": ctx.intelligence_result.get("motivations", []),
            "payment_status": ctx.payment_result.get("status"),
            "transaction_id": None,
            "provider": None,
            "recommendations": ctx.strategy_result.get("recommendations", []),
            "customer_message": customer_message, "sales_reply": ctx.sales_result.get("message"),
            "run_id": ctx.run_id, "customer_analysis": {**ctx.intelligence_result, "fraud_assessment": ctx.fraud_result, "financial_intelligence": ctx.financial_intelligence_result},
            "offer_recommendations": ctx.offer_recommendation_result.get("recommendations", [])
        })
        return ctx


# Backward-compatible class name retained for existing imports.
FinTechOrchestrator = TrainingCenterOrchestrator
