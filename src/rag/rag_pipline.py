"""
RAG pipeline for Alloyance
--------------------------
Responsibilities:
 - Retrieve reference data for Life Cycle Assessment (LCA)
 - Generate Executive Summary and Circularity Analysis via Gemini or fallback text
 - Serve as a clean interface for report_tech.py

Supports:
 - Gemini API (via google.generativeai)
 - Static fallback summaries if no API key or model response

Usage:
 - Ensure GEMINI_API_KEY is set in environment variables
"""

import os
import logging
from typing import Dict

# Optional: dotenv for local testing
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------
# Gemini model setup
# ------------------------------------------------------
def _get_model():
    """Return Gemini model if available."""
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai not installed.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


# ------------------------------------------------------
# Prompt Templates
# ------------------------------------------------------
EXECUTIVE_SUMMARY_PROMPT = """
You are an expert sustainability analyst preparing the *Executive Summary* section of a Life Cycle Assessment (LCA) report.

Context:
Material: {material}
Circularity Score: {circ_score}
Recycled Content: {recycled_content}
Reuse Potential: {reuse_potential}
Recovery Rate: {recovery_rate}

Task:
1. Summarize the overall environmental and circularity performance of the material.
2. Include 4 key parts:
   - Introduction
   - Key Metrics
   - Assessment
   - Recommendations (3–4 actionable suggestions)
3. Write in a professional and concise style, suitable for executive-level reports.
4. Keep it under 250 words.
"""

CIRCULARITY_ANALYSIS_PROMPT = """
You are a circular economy expert preparing the *Circularity Analysis* section of an LCA report.

Context:
Material: {material}
Circularity Score: {circ_score}
Recycled Content: {recycled_content}
Reuse Potential: {reuse_potential}
Recovery Rate: {recovery_rate}

Task:
1. Write 3 structured sections:
   - Material Flow
   - Circular Economy Indicators
   - Opportunities for Improvement
2. Discuss how recycled inputs, reuse, and recovery affect resource efficiency.
3. Keep it clear, factual, and action-oriented (~300–400 words).
"""


# ------------------------------------------------------
# Gemini LLM Call
# ------------------------------------------------------
def _call_gemini(prompt: str, label: str) -> str:
    """Call Gemini if available; otherwise return empty string to trigger fallback."""
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available; using fallback for %s", label)
        return ""
    try:
        logger.info("Generating %s using Gemini...", label)
        model = _get_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.exception("Gemini generation failed for %s: %s", label, e)
        return ""


# ------------------------------------------------------
# Fallback Static Templates
# ------------------------------------------------------
def _fallback_summary(material, circ_score, recycled_content, reuse_potential, recovery_rate):
    """Static text if Gemini is unavailable."""
    return (
        f"This Life Cycle Assessment evaluates the environmental and circularity performance of {material}. "
        f"The analysis indicates a Circularity Score of {circ_score}%, supported by {recycled_content}% recycled content, "
        f"a reuse potential of {reuse_potential}%, and a recovery rate of {recovery_rate}%. "
        f"\n\nCircularity Assessment:\nThe material demonstrates moderate circular potential, "
        f"with strong reuse and recycled input levels but room for improvement in end-of-life recovery. "
        f"\n\nRecommendations:\n"
        f"1. Increase post-use collection and recovery efficiency.\n"
        f"2. Integrate more secondary materials in production.\n"
        f"3. Implement design-for-reuse and modular strategies."
    )


def _fallback_circularity(material, circ_score, recycled_content, reuse_potential, recovery_rate):
    """Static text if Gemini is unavailable."""
    return (
        f"Material Flow:\nApproximately {recycled_content}% of {material} comes from recycled inputs, "
        f"reducing reliance on virgin extraction. The reuse potential of {reuse_potential}% helps extend "
        f"product lifecycles, while {recovery_rate}% of materials are currently recovered at end-of-life.\n\n"
        f"Circular Economy Indicators:\nThe Circularity Score of {circ_score}% indicates a balanced performance "
        f"across recycling, reuse, and recovery dimensions, though system inefficiencies still limit overall retention.\n\n"
        f"Opportunities for Improvement:\n"
        f"- Increase use of recycled feedstock and expand take-back systems.\n"
        f"- Improve recovery processes through better sorting and reprocessing.\n"
        f"- Promote product design strategies that facilitate disassembly and reuse."
    )


# ------------------------------------------------------
# Public API
# ------------------------------------------------------
def generate_summary(material: str, circ_score: float, recycled_content: float, reuse_potential: float, recovery_rate: float) -> str:
    """Generate Executive Summary for PDF report."""
    prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        material=material,
        circ_score=circ_score,
        recycled_content=recycled_content,
        reuse_potential=reuse_potential,
        recovery_rate=recovery_rate,
    )
    result = _call_gemini(prompt, "Executive Summary")
    return result or _fallback_summary(material, circ_score, recycled_content, reuse_potential, recovery_rate)


def generate_circularity_analysis(material: str, circ_score: float, recycled_content: float, reuse_potential: float, recovery_rate: float) -> str:
    """Generate Circularity Analysis for PDF report."""
    prompt = CIRCULARITY_ANALYSIS_PROMPT.format(
        material=material,
        circ_score=circ_score,
        recycled_content=recycled_content,
        reuse_potential=reuse_potential,
        recovery_rate=recovery_rate,
    )
    result = _call_gemini(prompt, "Circularity Analysis")
    return result or _fallback_circularity(material, circ_score, recycled_content, reuse_potential, recovery_rate)


# ------------------------------------------------------
# CLI test (optional)
# ------------------------------------------------------
if __name__ == "__main__":
    mat = "Copper Ore"
    circ, rec, reuse, recov = 52.8, 51.7, 60.2, 46.5
    print("\n=== EXECUTIVE SUMMARY ===\n")
    print(generate_summary(mat, circ, rec, reuse, recov))
    print("\n=== CIRCULARITY ANALYSIS ===\n")
    print(generate_circularity_analysis(mat, circ, rec, reuse, recov))
