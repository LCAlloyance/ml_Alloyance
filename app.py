from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
import random
import datetime
import os

class UserLogin(BaseModel):
    email: str
    password: str

app = FastAPI(title="Circular Metals API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ALL API ROUTES FIRST =====
#just for reference -these apis are not production ready and can be atatcked easily 
#before deploing need to check their security


@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/login")
async def login_user(user_login: UserLogin):
    if user_login.email.endswith("@gmail.com") and user_login.password:
        return {
            "success": True,
            "token": "real-backend-token-" + str(datetime.datetime.utcnow().timestamp()),
            "user": {"email": user_login.email},
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/assessment")
async def run_assessment(request: Request):
    payload = await request.json()
    recycled = float(payload.get("recycledContent", 50) or 0)
    
    base_circularity = 40 + (recycled * 0.45)
    if payload.get("energyUse") == "renewable": 
        base_circularity += 10
    if payload.get("transport") in ["local", "rail"]: 
        base_circularity += 5
    if payload.get("endOfLife") in ["recycle", "reuse"]: 
        base_circularity += 10

    environmental_score = 35 + (recycled * 0.5)
    if payload.get("energyUse") == "fossil": 
        environmental_score -= 15
    if payload.get("endOfLife") == "landfill": 
        environmental_score -= 20
    
    return {
        "circularityScore": max(0, min(100, round(base_circularity))),
        "environmentalScore": max(0, min(100, round(environmental_score))),
        "recommendations": random.sample([
            "Increase recycled content to reduce virgin input dependency.",
            "Optimize transport routes and prefer rail/sea logistics.",
            "Adopt closed-loop water systems in processing.",
            "Redesign product for easier disassembly and reuse.",
        ], 3),
        "missingParams": 0,
        "confidence": 95
    }

@app.get("/api/environmental-impact")
async def get_environmental_impact():
    return [
        {"name": "CO2 Emissions", "conventional": 850, "circular": 320},
        {"name": "Energy Use", "conventional": 1200, "circular": 680},
        {"name": "Water Use", "conventional": 400, "circular": 180},
        {"name": "Waste Gen.", "conventional": 200, "circular": 45},
    ]

@app.get("/api/circularity-indicators")
async def get_circularity_indicators():
    return [
        {"name": "Recycled Content", "value": 65, "target": 80},
        {"name": "Resource Efficiency", "value": 72, "target": 85},
        {"name": "Product Life Ext.", "value": 58, "target": 75},
        {"name": "Reuse Potential", "value": 43, "target": 60},
    ]

@app.get("/api/flow-data")
async def get_flow_data():
    return [
        {"stage": "Extraction", "material": 100, "recycled": 0},
        {"stage": "Processing", "material": 95, "recycled": 60},
        {"stage": "Manufacturing", "material": 90, "recycled": 85},
        {"stage": "Use", "material": 88, "recycled": 83},
        {"stage": "End-of-Life", "material": 25, "recycled": 75},
    ]

@app.get("/api/pie-data")
async def get_pie_data():
    return [
        {"name": "Recycled", "value": 45, "color": "#10b981"},
        {"name": "Virgin", "value": 35, "color": "#6366f1"},
        {"name": "Recovered", "value": 20, "color": "#f59e0b"},
    ]

@app.post("/api/reports/export")
async def export_report_csv():
    csv_content = "Metric,Conventional,Circular\n"
    csv_content += "CO2 Emissions,850,320\n"
    csv_content += "Energy Use,1200,680\n"
    csv_content += "Water Use,400,180\n"
    csv_content += "Waste Generation,200,45\n"
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=alloyance_report.csv"}
    )

# ===== STATIC FILES LAST (IMPORTANT!) =====
#these part serves the react app -make sure run buid first 
# or else white screen problem can appeaar
#--Ignore--


BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(_file_)), "my-app", "build")

if os.path.exists(BUILD_DIR):
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        
        file_path = os.path.join(BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(BUILD_DIR, "index.html"))

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)