import argparse
import pickle
import os
from model.core.recommender import FoodRecommender
from model.data.dataset import FoodDataset


def train_model():
    """เทรนโมเดลและบันทึกไว้ใช้ในภายหลัง"""
    print("เริ่มการเทรนโมเดล...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_file_foods = os.path.join(BASE_DIR, "model/data/raw/foods_10000.csv")
    path_file_ratings = os.path.join(BASE_DIR, "model/data/raw/ratings_10000.csv")
    # ตรวจสอบไฟล์ที่จำเป็น
    if not os.path.exists(path_file_foods):
        raise FileNotFoundError("❌ ไม่พบไฟล์ model/data/raw/foods.csv กรุณาสร้างไฟล์นี้ก่อนรันเทรนโมเดล")

    if not os.path.exists(path_file_ratings):
        raise FileNotFoundError("❌ ไม่พบไฟล์ model/data/raw/user_ratings.csv กรุณาสร้างไฟล์นี้ก่อนรันเทรนโมเดล")

    # โหลดข้อมูล
    dataset = FoodDataset(
        # foods_path=path_file_foods,
        ratings_path=path_file_ratings
    )
    # dataset = FoodDataset()

    # สร้างโมเดล
    recommender = FoodRecommender(dataset)

    # บันทึกโมเดล
    model_path = "data/models/food_rec_model.pkl"
    os.makedirs("data/models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(recommender, f)

    file_size_kb = os.path.getsize(model_path) / 1024
    print("เทรนโมเดลเสร็จสิ้น และบันทึกไว้ที่ data/models/food_rec_model.pkl")
    print(f"✅ โมเดลถูกบันทึกแล้ว (ขนาดประมาณ {file_size_kb:.2f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="เทรนโมเดลแนะนำอาหาร")
    args = parser.parse_args()
    train_model()