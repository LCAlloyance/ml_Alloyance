import argparse
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import os

# ✅ Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CHROMA_PATH = "chroma"

# 🌱 Narrative-style Prompts (No JSON!)
ANALYST_PROMPT = """
You are an expert LCA analyst.

Context:
{context}

Task:
1. Focus only on the actual environmental data found in the context.
2. If there is no quantitative data, make reasonable assumptions based on industry best practices and clearly label them as assumptions.
3. Identify and rank the **top three environmental hotspots** (e.g., greenhouse gas emissions, water use, waste generation).
4. Explain why each hotspot is significant in a clear narrative format.
5. Write in full sentences as if preparing the "Hotspots" section of a sustainability report.
"""

CIRCULARITY_PROMPT = """
You are a circular economy strategist.

Context:
{context}

Identified Hotspots:
{hotspots}

Task:
For each hotspot, propose exactly one **practical and actionable circularity improvement strategy**.
Explain why this strategy is effective and, where possible, reference a real-world case study.
Write as if preparing a section of a report titled "Circularity Recommendations," using professional and persuasive language.
"""

COMPLIANCE_PROMPT = """
You are a sustainability compliance officer.

Context:
{context}

Analysis & Recommendations:
{analysis_and_strategies}

Task:
Evaluate potential regulatory risks and opportunities based on EU CBAM, REACH, and other relevant environmental regulations.
Write in a clear, structured narrative format, including:
- Key regulatory risks and why they matter
- Compliance opportunities that could benefit the organization
- Strategic recommendations for staying ahead of regulations

Write as if preparing the "Compliance & Risk Assessment" section of a board report.
"""

DATA_ASSESSOR_PROMPT = """
You are a data quality analyst.

Context:
{context}

Task:
Identify what data is missing or incomplete to perform a full LCA.
Organize your response into short sections:
- "Missing Data": List key gaps (e.g., energy consumption data, process emissions, recycling rates)
- "Recommended Next Steps": Suggest what data to collect next and why it would improve decision-making

Write in plain English for a sustainability manager who is not an LCA expert.
"""

SCENARIO_PROMPT = """
You are a sustainability scenario modeler.

Context:
{context}

Proposed Strategies:
{strategies}

Task:
Describe how implementing these strategies could change the environmental results.
For each strategy, estimate whether the impact reduction is likely to be low, medium, or high and briefly explain why.
Write this as a "What-If Scenario Analysis" section of a report, in clear, narrative language.
"""

SUMMARY_PROMPT = """
You are an expert sustainability communicator.

Hotspots:
{hotspots}

Strategies:
{strategies}

Compliance & Risks:
{compliance}

Scenario Outcomes:
{scenarios}

Task:
Write an executive summary for a decision-maker.
Include:
- Three key insights from the analysis
- The most critical recommended actions
- An assessment of urgency (low, medium, or high)
- The business value of taking action (e.g., cost savings, compliance, reputation)

Write in a persuasive, easy-to-read style suitable for a CEO.
"""

def call_gemini(model, prompt, label="Response"):
    """Helper function to send prompt to Gemini and print result."""
    print(f"\n🔎 Asking Gemini: {label}...\n")
    response = model.generate_content(prompt)
    print(f"📄 {label}:\n")
    print(response.text)
    return response.text

def main():
    # 1️⃣ Parse input
    parser = argparse.ArgumentParser(description="Multi-agent RAG pipeline with Gemini.")
    parser.add_argument("query_text", nargs="?", help="Your question or search query.")
    args = parser.parse_args()
    query_text = args.query_text or input("\n💡 Enter your query: ")

    # 2️⃣ Load database
    print("\n🔍 Loading embeddings and database...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=4)
    results = [(doc, (score + 1) / 2) for doc, score in results]
    filtered_results = [(doc, score) for doc, score in results if score >= 0.3]

    if not filtered_results:
        print("\n⚠️ No good matches found in the database.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])

    # 3️⃣ Setup Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️ No Gemini API key found!")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # 🔄 Multi-Agent Pipeline
    data_assessor_prompt = ChatPromptTemplate.from_template(DATA_ASSESSOR_PROMPT).format(context=context_text)
    data_gaps = call_gemini(model, data_assessor_prompt, label="Data Quality Assessment")

    analyst_prompt = ChatPromptTemplate.from_template(ANALYST_PROMPT).format(context=context_text)
    hotspots = call_gemini(model, analyst_prompt, label="LCA Hotspot Analysis")

    circularity_prompt = ChatPromptTemplate.from_template(CIRCULARITY_PROMPT).format(context=context_text, hotspots=hotspots)
    strategies = call_gemini(model, circularity_prompt, label="Circularity Recommendations")

    scenario_prompt = ChatPromptTemplate.from_template(SCENARIO_PROMPT).format(context=context_text, strategies=strategies)
    scenarios = call_gemini(model, scenario_prompt, label="What-If Scenario Modeling")

    combined_analysis = f"Hotspots:\n{hotspots}\n\nStrategies:\n{strategies}\n\nScenarios:\n{scenarios}"
    compliance_prompt = ChatPromptTemplate.from_template(COMPLIANCE_PROMPT).format(context=context_text, analysis_and_strategies=combined_analysis)
    compliance_report = call_gemini(model, compliance_prompt, label="Compliance & Risk Assessment")

    summary_prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT).format(
        hotspots=hotspots, strategies=strategies, compliance=compliance_report, scenarios=scenarios
    )
    executive_summary = call_gemini(model, summary_prompt, label="Executive Summary")

    # 🏁 Final Report
    print("\n✅ MULTI-AGENT REPORT COMPLETE ✅\n")
    print("=== FINAL SUSTAINABILITY REPORT ===")
    print("\n📊 Data Gaps & Next Steps:\n", data_gaps)
    print("\n📈 Hotspot Analysis:\n", hotspots)
    print("\n♻️ Circularity Strategies:\n", strategies)
    print("\n🔮 Scenario Modeling:\n", scenarios)
    print("\n⚖️ Compliance & Risks:\n", compliance_report)
    print("\n📝 Executive Summary:\n", executive_summary)

if __name__ == "__main__":
    main()
