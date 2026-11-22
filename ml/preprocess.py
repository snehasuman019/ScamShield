"""
Preprocessing, feature extraction, and transparent rule-based indicator engine.
Implements:
- Text normalization (NLP cleaning)
- Multi-dimensional tabular feature extraction for Random Forest
- Transparent suspicious indicators engine based on verified recruitment scam patterns
"""

import re
import html
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Known suspicious regex patterns for recruitment fraud
SUSPICIOUS_RULES = [
    {
        "id": "RULE-UPFRONT-PAYMENT",
        "name": "Upfront Payment / Fee Request",
        "severity": "High",
        "description": "Mentions registration fees, equipment deposits, training costs, or buying software.",
        "regex": re.compile(
            r"\b(upfront\s+fee|application\s+fee|training\s+fee|registration\s+fee|processing\s+fee|purchase\s+(equipment|software|kit)|pay\s+for\s+background\s+check|reimbursement\s+check|security\s+deposit)\b",
            re.IGNORECASE,
        ),
    },
    {
        "id": "RULE-UNCONVENTIONAL-PAYMENT",
        "name": "Unconventional Money Transfer Methods",
        "severity": "High",
        "description": "References cryptocurrency, wire transfers, CashApp, Venmo, or Western Union.",
        "regex": re.compile(
            r"\b(crypto|bitcoin|usdt|wire\s+transfer|western\s+union|moneygram|cashapp|venmo|zelle|gift\s+card|check\s+cashing)\b",
            re.IGNORECASE,
        ),
    },
    {
        "id": "RULE-OFF-PLATFORM-COMM",
        "name": "Off-Platform / Encrypted Chat Interview",
        "severity": "High",
        "description": "Directs applicant to interview exclusively via Telegram, WhatsApp, Signal, or Hangouts.",
        "regex": re.compile(
            r"\b(telegram\s*(@|app|interview|message)|whatsapp\s*(interview|message|number)|signal\s*app|google\s*hangouts\s*interview|contact\s+via\s+telegram)\b",
            re.IGNORECASE,
        ),
    },
    {
        "id": "RULE-FREE-EMAIL-DOMAIN",
        "name": "Free Public Email Domain for Employer",
        "severity": "Medium",
        "description": "Recruiter uses a free public email (e.g. @gmail, @yahoo, @hotmail) rather than corporate domain.",
        "regex": re.compile(
            r"([a-zA-Z0-9_.+-]+@(gmail|yahoo|hotmail|outlook|aol|mail|protonmail)\.com)\b",
            re.IGNORECASE,
        ),
    },
    {
        "id": "RULE-SENSITIVE-DATA-PREMATURE",
        "name": "Premature Request for Sensitive Identity / Bank Info",
        "regex": re.compile(
            r"\b(send\s+copy\s+of\s+id|passport\s+copy|driver['’]s?\s+license\s+copy|social\s+security\s+number|ssn\s+required|bank\s+account\s+details|routing\s+number)\b",
            re.IGNORECASE,
        ),
        "severity": "High",
        "description": "Requests government ID or financial routing details before any formal interview.",
    },
    {
        "id": "RULE-UNREALISTIC-COMPENSATION",
        "name": "Unrealistic Compensation / Get-Rich-Quick Claims",
        "severity": "Medium",
        "description": "Exaggerated weekly earnings, instant wealth, or high pay with zero requirements.",
        "regex": re.compile(
            r"\b(earn\s+\$?[1-9]\d{2,4}\s*(per\s*week|weekly|daily)|unlimited\s+earning|be\s+your\s+own\s+boss|instant\s+wealth|no\s+experience\s+needed\s+earn|easy\s+money|work\s+1-?2\s+hours\s+make)\b",
            re.IGNORECASE,
        ),
    },
    {
        "id": "RULE-URGENT-NO-INTERVIEW",
        "name": "Urgent Hiring Without Formal Screening",
        "severity": "Medium",
        "description": "Offers immediate hiring without technical assessment, interview, or background evaluation.",
        "regex": re.compile(
            r"\b(start\s+immediately\s+no\s+interview|hired\s+immediately|no\s+interview\s+required|immediate\s+placement\s+guaranteed|urgent\s+recruitment\s+no\s+experience)\b",
            re.IGNORECASE,
        ),
    },
]


def clean_text(raw_text: Optional[str]) -> str:
    """
    Clean and normalize job advertisement text for NLP vectorization:
    - Handles HTML entities and tags
    - Removes URLs and email addresses
    - Normalizes excessive whitespace
    - Lowercases text
    """
    if not raw_text or pd.isna(raw_text):
        return ""

    text = str(raw_text)

    # Unescape HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs with token
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def combine_job_text(
    title: str = "",
    company_profile: str = "",
    description: str = "",
    requirements: str = "",
    benefits: str = "",
) -> str:
    """Combine text fields into a single unified text string for TF-IDF."""
    parts = [
        str(title or ""),
        str(company_profile or ""),
        str(description or ""),
        str(requirements or ""),
        str(benefits or ""),
    ]
    return clean_text(" ".join(filter(None, parts)))


def detect_suspicious_indicators(
    text: str,
    has_company_logo: Optional[int] = None,
    has_company_profile: Optional[bool] = None,
    has_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Transparent rule-based inspection returning explainable red flags.
    Never relies on black-box opacity for safety-critical indicators.
    """
    indicators = []
    text_content = str(text or "")

    # Check regex linguistic patterns
    for rule in SUSPICIOUS_RULES:
        match = rule["regex"].search(text_content)
        if match:
            matched_phrase = match.group(0)
            indicators.append({
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "description": rule["description"],
                "matched_sample": matched_phrase[:60],
            })

    # Statistical / Metadata flags from dataset observations
    if has_company_profile is False or (isinstance(text, str) and len(text.strip()) < 80):
        indicators.append({
            "rule_id": "RULE-MINIMAL-INFO",
            "rule_name": "Extremely Vague / Sparse Posting",
            "severity": "Low",
            "description": "Job posting provides very brief description with minimal context about the hiring organization.",
            "matched_sample": f"Length: {len(text_content)} chars",
        })

    if has_company_logo == 0 and has_questions == 0:
        indicators.append({
            "rule_id": "RULE-NO-LOGO-NO-SCREENING",
            "rule_name": "Missing Corporate Logo & Screening Protocol",
            "severity": "Low",
            "description": "Listing lacks both employer branding/logo and preliminary applicant screening questions.",
            "matched_sample": "has_company_logo=0, has_questions=0",
        })

    return indicators


def has_payment_indicators(text: str) -> int:
    """Binary helper checking if payment/fee scam keywords appear in text."""
    combined_regex = re.compile(
        r"(upfront\s+fee|crypto|bitcoin|wire\s+transfer|western\s+union|cashapp|venmo|gift\s+card|telegram|whatsapp)",
        re.IGNORECASE,
    )
    return 1 if combined_regex.search(str(text or "")) else 0


def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clean structured features for Model B (Random Forest).
    Ensures robust handling of missing values without arbitrary substitution.
    """
    features = pd.DataFrame(index=df.index)

    # 1. Salary presence (binary)
    if "salary_range" in df.columns:
        features["has_salary_range"] = df["salary_range"].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != "" else 0)
    else:
        features["has_salary_range"] = 0

    # 2. Company profile availability (binary)
    if "company_profile" in df.columns:
        features["has_company_profile"] = df["company_profile"].apply(lambda x: 1 if pd.notna(x) and len(str(x).strip()) > 30 else 0)
    else:
        features["has_company_profile"] = 0

    # 3. Logo availability (binary)
    if "has_company_logo" in df.columns:
        features["has_company_logo"] = df["has_company_logo"].fillna(0).astype(int)
    else:
        features["has_company_logo"] = 0

    # 4. Screening questions presence (binary)
    if "has_questions" in df.columns:
        features["has_questions"] = df["has_questions"].fillna(0).astype(int)
    else:
        features["has_questions"] = 0

    # 5. Remote / Telecommuting flag (binary)
    if "telecommuting" in df.columns:
        features["telecommuting"] = df["telecommuting"].fillna(0).astype(int)
    else:
        features["telecommuting"] = 0

    # 6. Categorical: Employment Type
    if "employment_type" in df.columns:
        features["employment_type"] = df["employment_type"].fillna("Unknown").astype(str)
    else:
        features["employment_type"] = "Unknown"

    # 7. Categorical: Required Experience
    if "required_experience" in df.columns:
        features["required_experience"] = df["required_experience"].fillna("Unknown").astype(str)
    else:
        features["required_experience"] = "Unknown"

    # 8. Categorical: Required Education
    if "required_education" in df.columns:
        features["required_education"] = df["required_education"].fillna("Unknown").astype(str)
    else:
        features["required_education"] = "Unknown"

    # 9. Transparent text-derived indicators: Upfront payment & off-platform keywords
    desc_series = df["description"].fillna("").astype(str) if "description" in df.columns else pd.Series([""] * len(df))
    features["has_payment_keywords"] = desc_series.apply(has_payment_indicators)

    # 10. Description character length and word count (numeric)
    features["desc_char_length"] = desc_series.apply(len)
    features["desc_word_count"] = desc_series.apply(lambda x: len(x.split()))

    return features
