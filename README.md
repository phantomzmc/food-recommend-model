# 🍜 Food Recommendation System

ระบบแนะนำอาหารโดยใช้ข้อมูลการให้คะแนนและลักษณะของอาหาร เพื่อสร้างโมเดลแนะนำอาหารที่เหมาะสมกับผู้ใช้แต่ละคน

## 📂 โครงสร้างโปรเจกต์
```markdown
food-recommend-model/
├── api/                     # ส่วนของ FastAPI สำหรับบริการ REST API
├── data/
│   ├── foods.csv            # ข้อมูลอาหาร (อัตโนมัติสร้างถ้าไม่มี)
│   ├── ratings.csv          # ข้อมูลการให้คะแนน (อัตโนมัติสร้างถ้าไม่มี)
│   └── models/
│       └── food_rec_model.pkl   # โมเดลที่ผ่านการฝึก
├── model/
│   ├── data/
│   │   └── dataset.py       # โมดูลโหลดข้อมูลอาหารและเรตติ้ง
│   └── recommend.py         # โมดูลคำนวณและแนะนำอาหาร
├── scripts/
│   └── train_model.py       # สคริปต์สำหรับเทรนโมเดล
└── README.md
```

## 🚀 วิธีใช้งาน

### 1. สร้าง Virtual Environment และติดตั้ง dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. เทรนโมเดล
```bash
python scripts/train_model.py
```
ระบบจะโหลดหรือสร้างข้อมูลอาหารและการให้คะแนน และฝึกโมเดล Collaborative Filtering ด้วย Cosine Similarity จากนั้นบันทึกไฟล์ไว้ใน data/models/food_rec_model.pkl

3. รัน API
```bash
cd api
uvicorn main:app --reload
```

สามารถเข้าผ่าน Swagger ได้ที่ http://localhost:8000/docs

```
📈 ฟีเจอร์
	•	แนะนำอาหารจากความชอบของผู้ใช้
	•	รับข้อมูลการให้คะแนนใหม่ และอัปเดตแบบเรียลไทม์
	•	รองรับการเทรนซ้ำเมื่อข้อมูลมีการเปลี่ยนแปลง
	•	API สำหรับการแนะนำอาหารใหม่

🧠 โมเดลที่ใช้
	•	User-based Collaborative Filtering โดยใช้ Cosine Similarity
	•	ข้อมูลถูกปรับให้อยู่ในช่วงเดียวกันด้วย MinMaxScaler ก่อนคำนวณ
	•	โมเดลถูก serialize ด้วย pickle

📊 ข้อมูลตัวอย่าง
	•	ข้อมูลอาหาร: foods.csv (มี 10,000 รายการโดยสุ่ม)
	•	ข้อมูลการให้คะแนน: ratings.csv (มี 10,000 รายการจากผู้ใช้หลากหลายคน)

```
🛠 ตัวอย่าง API

แนะนำอาหารให้ผู้ใช้

``
GET /recommendations/{user_id}?top_k=5
``

ให้คะแนนอาหาร

POST /rate
```json
{
  "user_id": 1,
  "food_id": 3,
  "rating": 4
}
```