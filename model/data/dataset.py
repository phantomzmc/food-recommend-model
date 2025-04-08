import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FoodDataset:
    def __init__(self, foods_path=None, ratings_path=None):
        print("📦 Initializing FoodDataset...")
        self.foods_df = None
        self.ratings_df = None
        self.load_data(foods_path, ratings_path)

    def load_data(self, foods_path=None, ratings_path=None):
        print("📥 Loading data...")
        """โหลดข้อมูลอาหารและการให้คะแนน"""
        # โหลดข้อมูลจากไฟล์หรือสร้างข้อมูลตัวอย่าง
        if foods_path:
            self.foods_df = pd.read_csv(foods_path)
        else:
            self.create_simple_foods()

        if ratings_path:
            self.ratings_df = pd.read_csv(ratings_path)
        else:
            self._create_sample_ratings()

    def create_simple_foods(self):
        print("🍽️ Creating sample food data...")
        """สร้างข้อมูลอาหารตัวอย่าง"""
        # โค้ดสร้างข้อมูลอาหารตัวอย่าง

        foods_data = {
            'food_id': range(1, 21),
            'name': [
                'ข้าวผัด', 'ผัดไทย', 'ต้มยำกุ้ง', 'ส้มตำ', 'แกงเขียวหวาน',
                'ผัดกระเพรา', 'ข้าวมันไก่', 'ก๋วยเตี๋ยวเรือ', 'หมูทอดกระเทียม', 'ข้าวหมูกระเทียม',
                'แกงมัสมั่น', 'แกงส้ม', 'ข้าวคลุกกะปิ', 'ข้าวต้มปลา', 'ผัดซีอิ๊ว',
                'ข้าวหน้าเป็ด', 'ไก่ทอด', 'กะเพราไก่ไข่ดาว', 'บะหมี่เกี๊ยว', 'ข้าวผัดปู'
            ],
            'spicy_level': [2, 3, 5, 5, 4, 1, 5, 4, 3, 2, 2, 1, 2, 1, 3, 1, 2, 3, 4, 3],
            'sweet_level': [2, 3, 2, 2, 3, 2, 1, 1, 4, 2, 1, 2, 2, 1, 2, 1, 2, 2, 3, 2],
            'salt_level': [3, 4, 4, 4, 3, 2, 3, 4, 3, 3, 3, 2, 3, 2, 4, 2, 3, 3, 4, 3],
            'price': [50, 60, 120, 40, 80, 50, 60, 70, 100, 60, 70, 60, 80, 40, 120, 40, 50, 90, 70, 60],
            'cooking_time': [10, 15, 25, 10, 30, 20, 10, 20, 40, 15, 15, 25, 20, 5, 30, 15, 20, 25, 15, 25],
            'category': [
                'อาหารจานเดียว', 'อาหารจานเดียว', 'แกง', 'ยำ', 'แกง',
                'อาหารจานเดียว', 'อาหารจานเดียว', 'อาหารจานเดียว', 'อาหารจานเดียว', 'อาหารจานเดียว',
                'แกง', 'แกง', 'อาหารจานเดียว', 'อาหารจานเดียว', 'อาหารจานเดียว',
                'อาหารจานเดียว', 'อาหารจานด่วน', 'อาหารจานด่วน', 'อาหารจานเดียว', 'อาหารจานเดียว'
            ]
        }
        self.foods_df = pd.DataFrame(foods_data)

    def _create_sample_ratings(self):
        print("⭐ Creating sample ratings data...")
        """สร้างข้อมูลการให้คะแนนตัวอย่าง"""
        ratings_data = {
            'user_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'food_id': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'rating': [5, 3, 4, 5, 3, 2, 4, 3, 2, 4]
        }
        self.ratings_df = pd.DataFrame(ratings_data)

    def get_food_features(self, features):
        """ดึงคุณลักษณะของอาหารสำหรับการคำนวณความคล้ายคลึง"""
        # Normalize คุณลักษณะ
        scaler = MinMaxScaler()
        return scaler.fit_transform(self.foods_df[features])

    def get_user_ratings(self, user_id):
        """ดึงการให้คะแนนของผู้ใช้"""
        if self.ratings_df is None:
            return pd.DataFrame()
        return self.ratings_df[self.ratings_df['user_id'] == user_id]

    def get_all_food_ids(self):
        """ดึงรหัสอาหารทั้งหมด"""
        return self.foods_df['food_id'].values

    def get_index_for_food_id(self, food_id):
        """ดึงดัชนีของอาหารจาก food_id"""
        return self.foods_df[self.foods_df['food_id'] == food_id].index[0]

    def get_food_details(self, food_ids):
        """ดึงรายละเอียดของอาหารตามรหัส"""
        result = []
        for food_id in food_ids:
            food = self.foods_df[self.foods_df['food_id'] == food_id].iloc[0]
            result.append({
                'id': int(food_id),
                'name': food['name'],
                'category': food['category'],
                'spicy_level': int(food['spicy_level']),
                'price': int(food['price'])
            })
        return result

    def get_random_foods(self, n):
        """สุ่มเลือกอาหาร n รายการ"""
        random_foods = self.foods_df.sample(n)
        result = []
        for _, food in random_foods.iterrows():
            result.append({
                'id': int(food['food_id']),
                'name': food['name'],
                'category': food['category'],
                'spicy_level': int(food['spicy_level']),
                'price': int(food['price'])
            })
        return result

    def add_rating(self, user_id, food_id, rating):
        print(f"📝 Adding or updating rating for user {user_id}, food {food_id} with score {rating}...")
        """เพิ่มหรืออัพเดตการให้คะแนน"""
        existing = self.ratings_df[
            (self.ratings_df['user_id'] == user_id) &
            (self.ratings_df['food_id'] == food_id)
            ]

        if len(existing) > 0:
            # อัพเดตคะแนน
            self.ratings_df.loc[
                (self.ratings_df['user_id'] == user_id) &
                (self.ratings_df['food_id'] == food_id),
                'rating'
            ] = rating
        else:
            # เพิ่มคะแนนใหม่
            new_rating = pd.DataFrame([{
                'user_id': user_id,
                'food_id': food_id,
                'rating': rating
            }])
            self.ratings_df = pd.concat([self.ratings_df, new_rating], ignore_index=True)
        return True