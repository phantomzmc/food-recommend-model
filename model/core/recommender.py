import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ..data.dataset import FoodDataset


class FoodRecommender:
    def __init__(self, dataset=None):
        self.dataset = dataset or FoodDataset()
        self.similarity_matrix = None
        self._compute_similarity()


    def _compute_similarity(self):
        """คำนวณความคล้ายคลึงระหว่างอาหาร"""
        features = ['spicy_level', 'sweet_level', 'salt_level', 'price', 'cooking_time']
        food_features = self.dataset.get_food_features(features)
        self.similarity_matrix = cosine_similarity(food_features)

    def recommend_for_user(self, user_id, n=5):
        """แนะนำอาหารสำหรับผู้ใช้"""
        user_ratings = self.dataset.get_user_ratings(user_id)

        if len(user_ratings) == 0:
            return self.random_recommendations(n)

        # คำนวณคะแนนแนะนำ
        all_food_ids = set(self.dataset.get_all_food_ids())
        rated_food_ids = set(user_ratings['food_id'])
        unrated_food_ids = all_food_ids - rated_food_ids

        food_scores = {}
        for unrated_id in unrated_food_ids:
            score = 0
            for _, row in user_ratings.iterrows():
                rated_id = row['food_id']
                rating = row['rating']

                # ดัชนีของอาหารในเมทริกซ์ความคล้ายคลึง
                rated_idx = self.dataset.get_index_for_food_id(rated_id)
                unrated_idx = self.dataset.get_index_for_food_id(unrated_id)

                similarity = self.similarity_matrix[rated_idx, unrated_idx]
                score += similarity * rating

            food_scores[unrated_id] = score

        # เรียงลำดับและเลือก n อันดับแรก
        sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
        top_foods = sorted_foods[:n]

        return self.dataset.get_food_details(food_ids=[food_id for food_id, _ in top_foods])

    def random_recommendations(self, n=5):
        """แนะนำอาหารแบบสุ่ม"""
        return self.dataset.get_random_foods(n)

    def recommend_by_meal(self, user_id, meal_time, n=5):
        """แนะนำอาหารตามมื้อสำหรับผู้ใช้"""
        # ดึงอาหารที่เหมาะกับมื้อที่ระบุ
        valid_food_ids = self.dataset.get_foods_by_meal(meal_time)
        if len(valid_food_ids) == 0:
            return self.random_recommendations_by_meal(meal_time, n)

        user_ratings = self.dataset.get_user_ratings(user_id)

        if len(user_ratings) == 0:
            return self.random_recommendations_by_meal(meal_time, n)

        # คำนวณคะแนนแนะนำเฉพาะอาหารที่เข้ากับมื้อ
        rated_food_ids = set(user_ratings['food_id'])
        unrated_food_ids = set(valid_food_ids) - rated_food_ids

        # ถ้าไม่พบอาหารมื้อนี้ที่ผู้ใช้ยังไม่ได้ให้คะแนน ก็ให้รวมอาหารที่เคยให้คะแนนด้วย
        if len(unrated_food_ids) < n:
            unrated_food_ids = unrated_food_ids.union(set(valid_food_ids).intersection(rated_food_ids))

        food_scores = {}
        for unrated_id in unrated_food_ids:
            score = 0
            for _, row in user_ratings.iterrows():
                rated_id = row['food_id']
                rating = row['rating']

                # ดัชนีของอาหารในเมทริกซ์ความคล้ายคลึง
                try:
                    rated_idx = self.dataset.get_index_for_food_id(rated_id)
                    unrated_idx = self.dataset.get_index_for_food_id(unrated_id)

                    similarity = self.similarity_matrix[rated_idx, unrated_idx]
                    score += similarity * rating
                except:
                    # ป้องกันข้อผิดพลาดถ้าข้อมูลไม่ตรงกัน
                    continue

            food_scores[unrated_id] = score

        # เรียงลำดับและเลือก n อันดับแรก
        sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
        top_foods = sorted_foods[:n]

        return self.dataset.get_food_details(food_ids=[food_id for food_id, _ in top_foods])

    def random_recommendations_by_meal(self, meal_time, n=5):
        """แนะนำอาหารแบบสุ่มตามมื้อ"""
        return self.dataset.get_random_foods_by_meal(meal_time, n)