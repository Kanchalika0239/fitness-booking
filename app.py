import streamlit as st
import pandas as pd
import re
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# ตั้งค่าหน้าต่างเว็บไซต์
st.set_page_config(page_title="ระบบจองฟิตเนสอัจฉริยะ", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ ระบบจองฟิตเนส (Smart Fitness Booking System)")
st.subheader("ระบบประมวลผลข้อความสำหรับการจองบริการฟิตเนส")
st.write("พิมพ์ข้อความแจ้งความต้องการจองคลาส จองโซนเล่น นัดเทรนเนอร์ หรือสอบถามข้อมูลได้เลย")

# 1. โหลดข้อมูลและโมเดล AI (ใช้ Caching เพื่อความเร็วในการโหลดเว็บ)
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    return df

@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

df = load_data()
model = load_model()

# คำนวณ Vector ของชุดข้อมูลเตรียมไว้
@st.cache_data
def get_dataset_embeddings(_df):
    return model.encode(_df['text'].tolist())

dataset_embeddings = get_dataset_embeddings(df)

# ฟังก์ชันทำความสะอาดข้อความ
def clean_text(text):
    text = re.sub(r'[^\w\s\n]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ฟังก์ชันตัดคำภาษาไทย
def tokenize_text(text):
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    return " ".join(tokens)

# ส่วนรับข้อมูลอินพุตจากผู้ใช้บนหน้าเว็บ
user_input = st.text_input("กรอกข้อความที่ต้องการสั่งการ:", placeholder="เช่น อยากลงเรียนคลาสปั่นจักรยานพรุ่งนี้เช้าครับ")

if st.button("ประมวลผลการจอง", type="primary"):
    if user_input.strip() != "":
        # กระบวนการ NLP
        cleaned_input = clean_text(user_input)
        tokenized_input = tokenize_text(cleaned_input)
        
        # คำนวณความคล้ายคลึงเชิงบริบทด้วย BERT
        user_vec = model.encode([cleaned_input])
        similarities = util.cos_sim(user_vec, dataset_embeddings)[0]
        
        best_match_idx = similarities.argmax().item()
        predicted_category = df.loc[best_match_idx, 'category']
        matched_text = df.loc[best_match_idx, 'text']
        confidence = similarities[best_match_idx].item()
        
        # แสดงผลลัพธ์บนหน้าเว็บ
        st.subheader("📌 ผลการจำแนกหมวดหมู่ความต้องการ")
        col1, col2 = st.columns(2)
        col1.metric("หมวดหมู่ที่จำแนกได้", predicted_category)
        col2.metric("ค่าความมั่นใจ (Confidence Score)", f"{confidence * 100:.2f}%")
        
        st.info(f"💡 ข้อความอ้างอิงที่ใกล้เคียงที่สุดในระบบ: '{matched_text}'")
        
        # แสดงรายละเอียดกระบวนการ NLP
        with st.expander("🔍 ดูรายละเอียดขั้นตอน NLP (Text Processing Pipeline)"):
            st.write("**1. ข้อความอินพุตดั้งเดิม:**", user_input)
            st.write("**2. ข้อความหลังทำความสะอาด (Regex & Cleansing):**", cleaned_input)
            st.write("**3. ผลการตัดคำภาษาไทย (Tokenization):**", tokenized_input)
    else:
        st.warning("กรุณากรอกข้อความก่อนกดประมวลผล")

# แสดงตัวอย่างชุดข้อมูลในระบบ
with st.expander("📊 ดูชุดข้อมูลทั้งหมดในระบบ (Dataset)"):
    st.dataframe(df)
