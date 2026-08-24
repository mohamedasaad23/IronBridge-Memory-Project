"""
Gemini (Google AI) client for Ironbridge agents.

Setup:
  1. Get a key: https://aistudio.google.com/apikey
  2. export GOOGLE_API_KEY=...   or put it in .env
  3. pip install google-generativeai python-dotenv

If no key is set, or quota is exhausted (429), falls back to offline replies
so the platform UI never shows raw Gemini errors to workers.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_raw_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
GOOGLE_API_KEY = (
    None
    if (not _raw_key or _raw_key in ("your-google-api-key-here", "changeme"))
    else _raw_key
)
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

_model = None
_import_failed = False


def available() -> bool:
    if _import_failed or not GOOGLE_API_KEY:
        return False
    return True


def _get_model():
    global _model, _import_failed
    if _model is not None:
        return _model
    if not GOOGLE_API_KEY or _import_failed:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        _model = genai.GenerativeModel(MODEL_NAME)
        return _model
    except Exception:
        _import_failed = True
        return None


SYSTEM_IRONBRIDGE = """You are the on-site AI assistant for Ironbridge Construction (Egypt).
You help workers with equipment, safety policy, certifications, and site procedures.
Rules:
- Answer in the same language the user uses (Arabic or English).
- Keep answers under 120 words unless they ask for detail.
- NEVER invent OSHA section numbers or policy text — only use the CONTEXT block.
- If CONTEXT is empty, say you don't have that policy loaded and suggest asking admin to add a RAG doc.
- If the user reports a broken machine / safety issue, say a maintenance ticket will be flagged.
- Do not approve deep trenches (>1.5m) yourself — tell them to use the High-Risk Dig agent / supervisor flow.
- Do not claim a worker's cert is valid unless CONTEXT says so.
"""


def chat(
    user_message: str,
    *,
    context: str = "",
    system: str = SYSTEM_IRONBRIDGE,
    max_tokens: int = 400,
    history: list | None = None,
) -> str:
    """Free-form constrained chat. Returns model text or offline fallback.
    `history` is optional prior turns: [{"role": "user"|"assistant", "content": str}, ...].
    """
    model = _get_model()
    if model is None:
        return _offline_chat(user_message, context)

    hist_block = ""
    if history:
        lines = []
        for turn in history[-8:]:
            role = (turn.get("role") or "user").upper()
            content = turn.get("content") or turn.get("text") or ""
            lines.append(f"{role}: {content}")
        hist_block = "\n".join(lines) + "\n\n"

    prompt = f"{system}\n\nCONTEXT:\n{context or '(none)'}\n\n{hist_block}USER:\n{user_message}"
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        text = (resp.text or "").strip()
        return text or _offline_chat(user_message, context)
    except Exception as e:
        # Never surface raw 429 / quota / network errors to the worker UI
        return _offline_chat(user_message, context)


def decide_json(
    user_message: str,
    schema_hint: str,
    *,
    context: str = "",
    system: str = "",
) -> dict[str, Any]:
    """
    Constrained decision: model must return JSON matching schema_hint.
    Used for routing / approve-reject style steps.
    """
    model = _get_model()
    sys = system or (
        "You are a strict JSON API. Reply with ONLY a single JSON object, no markdown."
    )
    prompt = (
        f"{sys}\n\nSchema: {schema_hint}\n\nCONTEXT:\n{context}\n\nUSER:\n{user_message}\n\nJSON:"
    )
    if model is None:
        return _offline_json(user_message)

    try:
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 300, "temperature": 0.1},
        )
        raw = (resp.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        return _offline_json(user_message)


def _offline_chat(msg: str, context: str) -> str:
    m = (msg or "").strip().lower()

    # Tasks — Arabic + English
    if any(w in m for w in ("مهام", "مهمتي", "مهامي", "tasks", "task", "النهاردة", "today")):
        if context and (
            "TASK" in context.upper()
            or "مهام" in context
            or "task" in context.lower()
        ):
            return "من بيانات الموقع (وضع محلي):\n" + context[:800]
        return (
            "وضع محلي (بدون نموذج حي). افتح تبويب المهام في الصفحة، "
            "أو اسأل الأدمن عن المهام المعيّنة لك. "
            "جرّب لاحقاً بعد استعادة حصة Gemini."
        )

    # Approval status
    if any(
        w in m
        for w in (
            "موافق",
            "موافقة",
            "approve",
            "approval",
            "تم الموافق",
            "هل تم",
        )
    ):
        return (
            "لا أقدر أؤكد موافقة من هنا بدون النظام الحي. "
            "طلبات المهندس تحتاج موافقة ADMIN1 و ADMIN2 من لوحة الأدمن. "
            "افتح Admin → Requests للاطلاع على الحالة."
        )

    # Attendance
    if any(w in m for w in ("حضور", "attendance", "clock", "سجّل", "سجل")):
        return (
            "لتسجيل الحضور: استخدم زر الحضور في الصفحة (Clock in). "
            "بعدها ينتظر موافقة الأدمن ليتحول من أحمر إلى أخضر."
        )

    # Policy — short Arabic summary, no noisy dumps
    if context and "POLICY" in context:
        return (
            "من سياسة الموقع (ملخص محلي): "
            "الخنادق أعمق من 1.5م تحتاج حماية وموافقة مشرف. "
            "الخوذة إلزامية. الحوادث شبه الفائتة تُبلَّغ خلال ساعتين."
        )

    if any(w in m for w in ("trench", "4.2b", "dig", "خندق", "حفر")):
        return (
            "الخنادق أعمق من 1.5م تحتاج نظام حماية وموافقة مشرف "
            "(وكيل الحفر عالي الخطورة)."
        )

    if "cert" in m or "شهاد" in m:
        return "تحقق من الشهادة عبر وكيل تجديد الشهادات (Cert Coordination)."

    return (
        "المساعد يعمل محلياً حالياً (انتهت حصة Gemini أو لا يوجد مفتاح). "
        "اسأل عن: مهامي، الحضور، أو سياسة السلامة — أو انتظر قليلاً وأعد المحاولة."
    )


def _offline_json(msg: str) -> dict[str, Any]:
    m = (msg or "").lower()
    if any(w in m for w in ("near-miss", "incident", "حادث")):
        return {"agent": "incident_handoff", "reason": "incident keywords"}
    if any(w in m for w in ("dig", "trench", "soil", "خندق", "حفر")):
        return {"agent": "high_risk_dig", "reason": "dig keywords"}
    if any(w in m for w in ("cert", "renew", "شهاد")):
        return {"agent": "cert_coordination", "reason": "cert keywords"}
    return {"agent": "memory_rag", "reason": "default"}
