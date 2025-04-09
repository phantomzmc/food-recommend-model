from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.params import Query
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

class RecommendRandomRequest(BaseModel):
    top_n: int = 5

class RecommendResponse(BaseModel):
    id: int
    name: str
    category: str
    spicy_level: int
    price: int
    meal_time: str

@app.get("/")
def root():
    return {"message": "Welcome to the Food Recommendation API!"}


@app.get("/foods/random", response_model=List[RecommendResponse])
def get_random_foods(n: int = 5):
    return dataset.get_random_foods(n)


@app.post("/ratings")
def submit_rating(request: RatingRequest):
    dataset.add_rating(request.user_id, request.food_id, request.rating)
    return {"message": "Rating submitted successfully"}


@app.post("/recommend", response_model=List[RecommendResponse])
def get_recommendation(request: RecommendRequest):
    try:
        recommended_ids = FoodRecommender().recommend_for_user(request.user_id, request.top_n)
        # food_details = dataset.get_food_details(recommended_ids)
        return {"user_id": request.user_id, "recommendations": recommended_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-random", response_model=List[RecommendResponse])
def get_recommendation_random(request: RecommendRandomRequest):
    try:
        recommended_ids = FoodRecommender().random_recommendations(request.top_n)
        # food_details = dataset.get_food_details(recommended_ids)
        return {"recommendations": recommended_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/by-meal", response_model=List[RecommendResponse])
def get_recommendation_by_meal(
    user_id:int = Query(1, description="User Id"),
    meal_time: str = Query("มื้อเช้า", description="มื้ออาหารที่ต้องการ (มื้อเช้า, มื้อกลางวัน, มื้อเย็น, มื้อค่ำ)"),
    n: int = Query(5, description="จำนวนคำแนะนำที่ต้องการ")
    ):
    """ขอคำแนะนำอาหารสำหรับผู้ใช้ตามมื้อ"""
    return recommender.recommend_by_meal(user_id, meal_time, n)