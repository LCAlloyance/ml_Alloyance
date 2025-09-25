import argparse
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

# ✅ Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 📊 Data Structure for Dynamic Input
@dataclass
class UserInputData:
    energy_source: str
    transport_mode: str
    transport_distance_km: float
    recycled_content_percent: float
    location: str
    functional_unit: str
    raw_material_type: str
    processing_method: str

@dataclass
class PredictedValues:
    gwp_kg_co2_eq: float
    material_circularity_indicator: float
    water_consumption_m3: float
    end_of_life_recycling_rate_percent: float
    energy_per_material_mj: float
    total_air_emissions_kg: float
    total_water_emissions_kg: float
    circularity_score: float
    potential_gwp_reduction_renewable_percent: Optional[float] = None
    potential_mci_improvement_recycling_percent: Optional[float] = None

@dataclass
class BenchmarkData:
    industry_average_gwp: float
    best_in_class_mci: float
    sector_average_water_m3: float

@dataclass
class DynamicContext:
    product_name: str
    process_route: str
    user_inputs: UserInputData
    ai_predictions: PredictedValues
    benchmarks: BenchmarkData
    
    def to_context_string(self) -> str:
        """Converts structured data into a narrative context string for the agents."""
        return f"""
SUSTAINABILITY ANALYSIS CONTEXT
==============================
Product: {self.product_name}
Process Route: {self.process_route}

USER-PROVIDED DATA:
- Energy Source: {self.user_inputs.energy_source}
- Transport: {self.user_inputs.transport_mode} ({self.user_inputs.transport_distance_km} km)
- Recycled Content Input: {self.user_inputs.recycled_content_percent}%
- Location: {self.user_inputs.location}
- Functional Unit: {self.user_inputs.functional_unit}
- Raw Material: {self.user_inputs.raw_material_type}
- Processing Method: {self.user_inputs.processing_method}

AI-PREDICTED PERFORMANCE METRICS:
- Global Warming Potential: {self.ai_predictions.gwp_kg_co2_eq} kg CO2-eq
- Material Circularity Indicator: {self.ai_predictions.material_circularity_indicator}
- Water Consumption: {self.ai_predictions.water_consumption_m3} m³
- End-of-Life Recycling Rate: {self.ai_predictions.end_of_life_recycling_rate_percent}%
- Energy Intensity: {self.ai_predictions.energy_per_material_mj} MJ per unit
- Total Air Emissions: {self.ai_predictions.total_air_emissions_kg} kg
- Total Water Emissions: {self.ai_predictions.total_water_emissions_kg} kg
- Overall Circularity Score: {self.ai_predictions.circularity_score}

IMPROVEMENT POTENTIALS:
- GWP Reduction (Renewable Energy): {self.ai_predictions.potential_gwp_reduction_renewable_percent or 'Not calculated'}%
- MCI Improvement (Increased Recycling): {self.ai_predictions.potential_mci_improvement_recycling_percent or 'Not calculated'}%

INDUSTRY BENCHMARKS:
- Industry Average GWP: {self.benchmarks.industry_average_gwp} kg CO2-eq
- Best-in-Class MCI: {self.benchmarks.best_in_class_mci}
- Sector Average Water Use: {self.benchmarks.sector_average_water_m3} m³
"""

# 🌱 Narrative-style Prompts (No JSON!) - No changes needed here.
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
    parser.add_argument("json_file", help="Path to the JSON input file with user data and predictions.")
    args = parser.parse_args()
    json_file_path = args.json_file

    # 2️⃣ Load JSON input and create context
    try:
        with open(json_file_path, 'r') as f:
            json_input = json.load(f)
        
        user_inputs = UserInputData(**json_input['user_inputs'])
        ai_predictions = PredictedValues(**json_input['ai_predictions'])
        benchmarks = BenchmarkData(**json_input['benchmarks'])
        
        dynamic_data = DynamicContext(
            product_name=json_input['product_name'],
            process_route=json_input['process_route'],
            user_inputs=user_inputs,
            ai_predictions=ai_predictions,
            benchmarks=benchmarks
        )
        
        # This is the single, dynamic context string for all agents
        context_text = dynamic_data.to_context_string()

        # You can optionally still use your local database to add static context
        # to the dynamic context if needed. Just uncomment these lines:
        # from langchain_huggingface import HuggingFaceEmbeddings
        # from langchain_chroma import Chroma
        # CHROMA_PATH = "chroma"
        # embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        # query_text = "sustainability report for primary aluminium production"
        # results = db.similarity_search_with_relevance_scores(query_text, k=4)
        # filtered_results = [(doc, score) for doc, score in results if score >= 0.3]
        # if filtered_results:
        #     static_context = "\n\n--- KNOWLEDGE BASE CONTEXT ---\n\n" + "\n\n".join([doc.page_content for doc, _ in filtered_results])
        #     context_text += static_context
        
    except FileNotFoundError:
        print(f"❌ Error: JSON file not found at {json_file_path}")
        return
    except (KeyError, TypeError) as e:
        print(f"❌ Error: Invalid data structure in JSON input file: {e}")
        return

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