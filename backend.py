import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import numpy as np

app = FastAPI()

# --- 允许跨域 ---
origins = ["http://localhost:8501", "http://127.0.0.1:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
try:
    model_new = keras.models.load_model('mnist.hdf5')
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    raise e

@app.post("/predict_json")
async def predict_json(item: dict):
    try:
        # 1. 获取数据
        data = np.array(item['data'], dtype='float32')
        
        # 2. 重塑形状 (1, 28, 28, 1)
        # 如果你的模型是 channels_first (Theano)，请改为 (1, 1, Shift, 28)
        input_data = data.reshape(1, 28, 28, 1)

        # 3. 预测 (注意：这里假设模型输入期望是 0-1)
        # 如果模型是 0-255 训练的，取消下面这行的注释
        # input_data = input_data * 255.0 
        
        prediction = model_new.predict(input_data, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))

        return {
            "result": predicted_class,
            "percent": prediction[0].tolist()
        }

    except Exception as e:
        print(f"预测出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 正在启动后端服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)