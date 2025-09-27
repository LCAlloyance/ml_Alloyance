from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any

# Import your modules (same as before)
from autofill import AutofillManager
from predict import PredictionEngine
from rag.rag_pipline import run_pipeline
from report_tech import generate_report_from_json

app = FastAPI(title="Sustainability Analysis API", version="1.0.0")


class SustainabilityEvaluator:
    """Main orchestrator for the sustainability analysis workflow"""

    def __init__(self):
        self.autofill_manager = AutofillManager()
        self.prediction_engine = PredictionEngine()
        self.results = {}

    def process_request(self, input_data: Dict, output_dir: str = "outputs") -> Dict[str, Any]:
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Step 1: Save input JSON
            input_json_path = os.path.join(output_dir, "input_data.json")
            with open(input_json_path, "w") as f:
                json.dump(input_data, f, indent=2)

            # Step 2: Auto-fill missing data
            enhanced_data = self.autofill_manager.enhance_data(input_data)
            enhanced_json_path = os.path.join(output_dir, "enhanced_data.json")
            with open(enhanced_json_path, "w") as f:
                json.dump(enhanced_data, f, indent=2)

            # Step 3: AI predictions
            predictions = self.prediction_engine.generate_predictions(enhanced_data)
            complete_context = self._build_complete_context(enhanced_data, predictions)
            context_json_path = os.path.join(output_dir, "complete_context.json")
            with open(context_json_path, "w") as f:
                json.dump(complete_context, f, indent=2)

            # Step 4: Run RAG pipeline
            rag_results = run_pipeline(context_json_path)
            rag_results_path = os.path.join(output_dir, "rag_analysis.json")
            with open(rag_results_path, "w") as f:
                json.dump(rag_results, f, indent=2)

            # Step 5: Generate report
            report_path = os.path.join(output_dir, "sustainability_report.pdf")
            generate_report_from_json(context_json_path, report_path)

            self.results = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "files_generated": {
                    "enhanced_data": enhanced_json_path,
                    "complete_context": context_json_path,
                    "rag_analysis": rag_results_path,
                    "final_report": report_path
                },
                "analysis_results": rag_results,
                "predictions": predictions,
                "product_info": {
                    "name": complete_context.get("product_name"),
                    "process_route": complete_context.get("process_route")
                }
            }

            return self.results

        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }

    def _build_complete_context(self, enhanced_data: Dict, predictions: Dict) -> Dict:
        user_inputs = enhanced_data.get("user_inputs", {})
        return {
            "product_name": enhanced_data.get("product_name", "Unknown Product"),
            "process_route": enhanced_data.get("process_route", "Standard Process"),
            "user_inputs": {
                "energy_source": user_inputs.get("energy_source", "Mixed Grid"),
                "transport_mode": user_inputs.get("transport_mode", "Road"),
                "transport_distance_km": user_inputs.get("transport_distance_km", 100.0),
                "recycled_content_percent": user_inputs.get("recycled_content_percent", 0.0),
                "location": user_inputs.get("location", "Global"),
                "functional_unit": user_inputs.get("functional_unit", "1 unit"),
                "raw_material_type": user_inputs.get("raw_material_type", "Standard"),
                "processing_method": user_inputs.get("processing_method", "Conventional"),
            },
            "ai_predictions": predictions,
            "benchmarks": enhanced_data.get(
                "benchmarks",
                {"industry_average_gwp": 1000.0, "best_in_class_mci": 0.8, "sector_average_water_m3": 10.0},
            ),
        }


@app.post("/process")
async def process(input_json: UploadFile):
    """
    Accepts a JSON file from the frontend, runs the full pipeline,
    and returns results (with file paths).
    """
    try:
        contents = await input_json.read()
        data = json.loads(contents)

        evaluator = SustainabilityEvaluator()
        results = evaluator.process_request(data)

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            },
            status_code=500,
        )


@app.get("/download-report")
async def download_report(path: str):
    """
    Allows downloading the generated report by passing its path.
    """
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Report not found"}, status_code=404)
    return FileResponse(path, media_type="application/pdf", filename="sustainability_report.pdf")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
