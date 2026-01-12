from ultralytics import YOLO
import cv2
from fastapi import FastAPI,UploadFile,File
from fastapi.responses import StreamingResponse,FileResponse
import numpy as np
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

model=YOLO("weights/best.pt")
def detect(frame):
   results=model(frame,imgsz=640,conf=0.1)
   if results[0].boxes is None or len(results[0].boxes) == 0:
       print("❌ NO DETECTIONS")
   else:
       print("✅ DETECTIONS FOUND:", len(results[0].boxes))
       print("Classes:", results[0].boxes.cls.cpu().numpy())
       print("Confidences:", results[0].boxes.conf.cpu().numpy())
   annotated=results[0].plot()
   return annotated

app=FastAPI(title="Drone Detection System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return FileResponse("frontend/index.html")

@app.get("/image")
def image_page():
    return FileResponse("frontend/image.html")

@app.get("/video")
def video_page():
    return FileResponse("frontend/video.html")

@app.post("/predict/image")
async def predict_image(file:UploadFile=File(...)):
   img=Image.open(file.file).convert('RGB')
   frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
   results=detect(frame)
   _,buffer=cv2.imencode(".jpg",results)
   return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )

@app.post("/predict/video")
async def predict_video(file:UploadFile=File(...)):
   temp_path=f"temp_{file.filename}"
   with open(temp_path, "wb") as f:
      f.write(await file.read())
   cap=cv2.VideoCapture(temp_path)
   out_path=f"out_{file.filename}"
   fourcc=cv2.VideoWriter_fourcc(*"mp4v")
   fps=cap.get(cv2.CAP_PROP_FPS)
   if fps == 0:
      fps = 25
   out=cv2.VideoWriter(out_path,fourcc,fps,(640,640))
   while cap.isOpened():
      ret,frame=cap.read()
      if not ret:
         break
      results=detect(frame)
      out.write(results)
   cap.release()
   out.release()
   return {"output_video": out_path}

