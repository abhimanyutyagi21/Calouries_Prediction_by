from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import joblib
import os
from typing import Optional

app = FastAPI()

# ---------------------------
#   CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
#   STATIC DIR + FRONTEND FILE
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index1.html")

# Ensure static folder exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static router
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return Response("index1.html NOT FOUND inside /static", status_code=404)

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# ---------------------------
#   LOAD MODEL
# ---------------------------
MODEL_FILE = os.path.join(BASE_DIR, "calories_predictor_model.joblib")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Model file missing: calories_predictor_model.joblib")

model = joblib.load(MODEL_FILE)

# ---------------------------
#   AUTH (FAKE)
# ---------------------------
fake_users_db = {}

class AuthRequest(BaseModel):
    name: Optional[str] = None
    email: str
    password: str

@app.post("/signup")
def signup(data: AuthRequest):
    if data.email in fake_users_db:
        return {"error": "User already exists"}
    fake_users_db[data.email] = {"name": data.name, "password": data.password}
    return {"message": "Signup successful"}

@app.post("/login")
def login(data: AuthRequest):
    user = fake_users_db.get(data.email)
    if not user or user["password"] != data.password:
        return {"error": "Invalid credentials"}
    return {"message": "Login successful"}

# ---------------------------
#   PREDICT REQUEST MODEL
# ---------------------------
class PredictRequest(BaseModel):
    age: Optional[float] = None
    gender: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    fat_percentage: Optional[float] = None
    bmi_input: Optional[float] = None

    workout_type: Optional[str] = None
    session_duration: Optional[float] = None
    workout_frequency: Optional[float] = None
    max_bpm: Optional[float] = None
    avg_bpm: Optional[float] = None
    resting_bpm: Optional[float] = None
    difficulty: Optional[str] = None
    body_part: Optional[str] = None

    meal_type: Optional[str] = None
    diet_type: Optional[str] = None
    cooking_method: Optional[str] = None
    water_intake: Optional[float] = None
    meals_frequency: Optional[float] = None
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    serving_size_g: Optional[float] = None
    prep_time_min: Optional[float] = None
    cook_time_min: Optional[float] = None
    rating: Optional[float] = None

    experience_level: Optional[float] = None
    physical_exercise: Optional[float] = None

# ---------------------------
#   CATEGORY ENCODER
# ---------------------------
def encode(category, mapping):
    if category is None:
        return 0
    return mapping.get(str(category).lower(), 0)

gender_map = {"male": 1, "female": 2, "other": 3}
workout_map = {"strength": 1, "cardio": 2, "flexibility": 3, "other": 4}
difficulty_map = {"beginner": 1, "intermediate": 2, "advanced": 3}
body_part_map = {"core": 1, "upper body": 2, "lower body": 3, "full body": 4, "other": 5}
meal_map = {"breakfast": 1, "lunch": 2, "dinner": 3, "snack": 4}
diet_map = {"standard": 1, "vegan": 2, "keto": 3, "paleo": 4, "mediterranean": 5}
cooking_map = {"baked": 1, "boiled": 2, "grilled": 3, "fried": 4, "steamed": 5}

# ---------------------------
#   PREDICT ENDPOINT
# ---------------------------
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        gender = encode(data.gender, gender_map)
        workout_type = encode(data.workout_type, workout_map)
        difficulty = encode(data.difficulty, difficulty_map)
        body_part = encode(data.body_part, body_part_map)
        meal_type = encode(data.meal_type, meal_map)
        diet_type = encode(data.diet_type, diet_map)
        cooking_method = encode(data.cooking_method, cooking_map)

        features = [
            data.age or 0, gender, data.weight or 0, data.height or 0,
            data.fat_percentage or 0, data.bmi_input or 0,
            workout_type, data.session_duration or 0,
            data.workout_frequency or 0, data.max_bpm or 0,
            data.avg_bpm or 0, data.resting_bpm or 0,
            difficulty, body_part,
            meal_type, diet_type, cooking_method,
            data.water_intake or 0, data.meals_frequency or 0,
            data.sugar_g or 0, data.sodium_mg or 0,
            data.cholesterol_mg or 0, data.serving_size_g or 0,
            data.prep_time_min or 0, data.cook_time_min or 0,
            data.rating or 0,
            data.experience_level or 0, data.physical_exercise or 0
        ]

        pred = model.predict([features])[0]

        return {"predicted_calories": float(pred)}

    except Exception as e:
        return {"error": str(e)}

# ---------------------------
#   HEALTH CHECK
# ---------------------------
@app.get("/status")
def health():
    return {"status": "FitTrack AI Backend Running!"}
