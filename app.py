# app.py
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------
# 모델 및 스케일러 초기화 (캐싱)
# --------------------------
@st.cache_resource
def create_model():
    X = np.random.rand(100, 15)
    y = np.random.choice(['rock_collision', 'sand_slip', 'system_noise'], size=100)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = create_model()

# --------------------------
# Streamlit 앱 시작
# --------------------------
st.set_page_config(page_title="우주 로버 소리 인식", layout="centered")
st.title("🚀 우주 로버 소리 기반 자율 상황 인식 시스템")
st.markdown("진동/소리 `.wav` 파일을 분석하여 로버 주변 상황을 예측합니다.")

# --------------------------
# 파일 업로드
# --------------------------
uploaded_file = st.file_uploader("🔊 WAV 파일 업로드", type=["wav"])

# --------------------------
# 특징 추출 함수
# --------------------------
def extract_features(file):
    y, sr = librosa.load(file, sr=16000)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.zero_crossings(y))
    return np.hstack([mfccs, centroid, zcr])

# --------------------------
# 분석 및 출력
# --------------------------
if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("🔍 소리 분석 중..."):
        features = extract_features(uploaded_file)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled).max()

        st.subheader("🧠 예측 결과")
        st.success(f"예측 상황: **{prediction}**  \n신뢰도: **{confidence:.2%}**")

        # 스펙트로그램 시각화
        y, sr = librosa.load(uploaded_file, sr=16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)

        st.subheader("📊 스펙트로그램")
        fig, ax = plt.subplots()
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(fig)
else:
    st.info("왼쪽 상단에서 `.wav` 파일을 업로드해 주세요.")
