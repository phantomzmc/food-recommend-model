from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys

# เพิ่ม path สำหรับ import dataset และ model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.data.dataset import FoodDataset
from model.core.recommender import FoodRecommender


app = FastAPI(title="🍽️ Food Recommendation API")

# โหลด Dataset และ Model
dataset = FoodDataset("model/data/raw/foods_10000.csv", "model/data/raw/ratings_10000.csv")
model_path = "scripts/data/models/food_rec_model.pkl"
recommender = joblib.load(model_path)


# สำหรับรับข้อมูลจาก frontend
class RatingRequest(BaseModel):
    user_id: int
    food_id: int
    rating: int


class RecommendRequest(BaseModel):
    user_id: int
    top_n: int = 5


@app.get("/")
def root():
    return {"message": "Welcome to the Food Recommendation API!"}


@app.get("/foods/random")
def get_random_foods(n: int = 5):
    return dataset.get_random_foods(n)


@app.post("/ratings")
def submit_rating(request: RatingRequest):
    dataset.add_rating(request.user_id, request.food_id, request.rating)
    return {"message": "Rating submitted successfully"}


@app.post("/recommend")
def get_recommendation(request: RecommendRequest):
    try:
        recommended_ids = FoodRecommender().recommend_for_user(request.user_id, request.top_n)
        # food_details = dataset.get_food_details(recommended_ids)
        return {"user_id": request.user_id, "recommendations": recommended_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))