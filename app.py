# app.py (完整版)

import streamlit as st
import subprocess
import sys
import time
import os

# --- 关键：补全所有必要的库引用 ---
import cv2
import numpy as np
import requests
from streamlit_drawable_canvas import st_canvas

# --- 1. 启动后端服务 (FastAPI) ---
def start_backend():
    # 使用 uvicorn 模块直接启动 backend.py 中的 app 实例
    # 这种方式比 if __name__ == "__main__" 更稳定
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "backend:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
    return backend_process

# --- 2. 初始化检查与启动 ---
if 'backend_started' not in st.session_state:
    with st.spinner("🚀 正在启动 AI 引擎..."):
        p = start_backend()
        st.session_state.backend_started = True
        st.session_state.backend_process = p
        # 给后端几秒钟时间加载模型
        time.sleep(3)

# ================== [前端界面代码开始] ==================

# --- 1. 页面配置与高级 CSS 样式 ---
st.set_page_config(page_title="MNIST Digit Predictor", page_icon="✏️")

# 注入自定义 CSS 以增强视觉效果
st.markdown("""
    <style>
        /* 全局背景与字体 */
        body { background-color: #f8f9fa; }
        /* 隐藏默认页眉和页脚 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        /* 标题区域样式 */
        .main-header { text-align: center; margin-bottom: 2rem; padding-top: 1rem; }
        .main-title { font-size: 3.5rem; font-weight: 800; color: #2c3e50; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: center; gap: 15px; }
        .sub-title { font-size: 1.2rem; color: #7f8c8d; font-weight: 400; }
        /* 画板容器美化 */
        .canvas-container { border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); overflow: hidden; background: white; padding: 10px; margin: 0 auto; max-width: 400px; }
        /* 按钮组样式 */
        div.row-widget.stButton > button { height: 3.5rem; font-size: 1.2rem; border-radius: 12px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        div.row-widget.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
        /* 结果展示卡片 */
        .result-card { background: white; border-radius: 16px; padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; margin-top: 2rem; border: 1px solid #eee; }
        .prediction-number { font-size: 5rem; font-weight: bold; color: #ff4b4b; line-height: 1; margin: 10px 0; }
        .confidence-label { font-size: 1rem; color: #95a5a6; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. 头部区域 ---
st.markdown("""
    <div class="main-header">
        <div class="main-title">✏️ MNIST Digit Predictor</div>
        <div class="sub-title">Draw a digit (0-9) and the CNN predicts it.</div>
    </div>
""", unsafe_allow_html=True)

# --- 3. 核心功能区 ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

# --- 4. 逻辑处理与结果显示 ---
if predict_btn and canvas_result.image_data is not None:
    img = canvas_result.image_data.astype('uint8')

    # --- 预处理逻辑 ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    _, binary_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(binary_img, (28, 28), interpolation=cv2.INTER_AREA)
    final_input = resized.astype('float32') / 255.0

    payload = {"data": final_input.flatten().tolist()}

    try:
        # 注意：在本地是 localhost，在云端也是 localhost (因为都在同一个容器内)
        response = requests.post("http://localhost:8000/predict_json", json=payload)
        data = response.json()

        predicted_digit = data.get('result', '?')
        confidence = data.get('confidence', 0)

        st.markdown(f"""
            <div class="result-card">
                <div class="confidence-label">Model Prediction</div>
                <div class="prediction-number">{predicted_digit}</div>
                <div style="margin-top: 20px;">
                    <p style="color: #7f8c8d; margin-bottom: 5px;">Confidence Score</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.progress(confidence, text=f"Accuracy: {confidence * 100:.2f}%")

        if 'percent' in data:
            st.caption("Probability Distribution:")
            st.bar_chart(data['percent'])

    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
        st.info("Backend might still be loading the model. Please wait a moment.")

elif clear_btn:
    st.rerun()

elif canvas_result.image_data is None:
    st.info("👆 Please draw a digit in the black box above.")