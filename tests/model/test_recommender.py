# tests/model/test_recommender.py
import pytest
import pandas as pd
from model.core.recommender import FoodRecommender
from model.data.dataset import FoodDataset


class TestFoodDataset:
    def test_load_data(self):
        """ทดสอบการโหลดข้อมูล"""
        dataset = FoodDataset()
        assert dataset.foods_df is not None
        assert dataset.ratings_df is not None
        assert len(dataset.foods_df) > 0
        assert len(dataset.ratings_df) > 0

    def test_get_user_ratings(self):
        """ทดสอบการดึงข้อมูลการให้คะแนนของผู้ใช้"""
        dataset = FoodDataset()
        user_ratings = dataset.get_user_ratings(1)
        assert len(user_ratings) > 0
        assert 'food_id' in user_ratings.columns
        assert 'rating' in user_ratings.columns

    def test_add_rating(self):
        """ทดสอบการเพิ่มคะแนน"""
        dataset = FoodDataset()
        initial_len = len(dataset.ratings_df)

        # เพิ่มการให้คะแนนใหม่
        dataset.add_rating(user_id=2, food_id=1, rating=4)

        # ตรวจสอบว่าข้อมูลเพิ่มขึ้น
        assert len(dataset.ratings_df) == initial_len + 1

        # อัพเดตคะแนนที่มีอยู่
        dataset.add_rating(user_id=2, food_id=1, rating=5)

        # ตรวจสอบว่าข้อมูลไม่เพิ่ม แต่ค่าเปลี่ยน
        assert len(dataset.ratings_df) == initial_len + 1
        assert dataset.ratings_df[
                   (dataset.ratings_df['user_id'] == 2) &
                   (dataset.ratings_df['food_id'] == 1)
                   ]['rating'].values[0] == 5


class TestFoodRecommender:
    def test_recommend_for_user(self):
        """ทดสอบการแนะนำอาหารสำหรับผู้ใช้"""
        dataset = FoodDataset()
        recommender = FoodRecommender(dataset)

        # ทดสอบการแนะนำ
        recommendations = recommender.recommend_for_user(user_id=1, n=3)

        # ตรวจสอบผลลัพธ์
        assert len(recommendations) == 3
        assert 'id' in recommendations[0]
        assert 'name' in recommendations[0]
        assert 'category' in recommendations[0]

    def test_random_recommendations(self):
        """ทดสอบการแนะนำอาหารแบบสุ่ม"""
        dataset = FoodDataset()
        recommender = FoodRecommender(dataset)

        # ทดสอบการแนะนำแบบสุ่ม
        recommendations = recommender.random_recommendations(n=5)

        # ตรวจสอบผลลัพธ์
        assert len(recommendations) == 5
        assert 'id' in recommendations[0]
        assert 'name' in recommendations[0]