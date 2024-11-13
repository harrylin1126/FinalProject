import asyncio
import datetime
import os
import json
import base64
import cv2
import numpy as np
import pandas as pd
import csv
import subprocess
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from websockets.server import serve

IPADDR="localhost"
PORT=9999
VIDEO_PATH=r"../baseball/src/Component/video/video.mp4"
TEXT_PATH = r"./yololabel.txt"
LastFrame_Path = r"./last_frame.jpg"
OutPut_Path = r"./output.jpg"
UnPredicted_Path = r"C:\FinalProject\server\balltype.csv"
DataSet_Path = r"C:\FinalProject\server\svmFull.csv"
class Model:
    
    def __init__(self, image_width=1920, image_height=1080, num_points=10):
        self.image_width = image_width
        self.image_height = image_height
        self.num_points = num_points
        self.svm_classifier = None

    def save_last_frame(self, video_path, output_image_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("無法開啟影片")
            return False
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(output_image_path, frame)
            print("已成功保存最後一幀為圖片")
            saved = True
        else:
            print("無法讀取最後一幀")
            saved = False

        cap.release()
        return saved

    def draw_coordinates_on_image(self, txt_file_path, image_path, output_image_path):
        coordinates = []
        try:
            with open(txt_file_path, "r", encoding="utf-8") as file:
                for line in file:
                    x, y = map(float, line.strip().split())
                    coordinates.append((int(x ), int(y)))
        except FileNotFoundError:
            print(f"無法找到檔案: {txt_file_path}")
            return False
        except ValueError:
            print("檔案格式錯誤，請確認每一行包含兩個數值座標")
            return False

        if not coordinates:
            print("無法讀取座標")
            return False

        points = np.array(coordinates, dtype=np.int32)
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖片: {image_path}")
            return False

        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=False, color=(0, 255, 0), thickness=5)
        cv2.imwrite(output_image_path, image)
        print("已成功保存帶有座標曲線的圖片")
        return True

    def process_and_save_coordinates(self, txt_file_path, output_csv_path):
        coordinates = []
        try:
            with open(txt_file_path, "r", encoding="utf-8") as file:
                for line in file:
                    x, y = map(float, line.strip().split())
                    coordinates.append((x , y ))
        except FileNotFoundError:
            print(f"無法找到檔案: {txt_file_path}")
            return False
        except ValueError:
            print("檔案格式錯誤，請確認每一行包含兩個數值座標")
            return False

        points = np.array(coordinates, dtype=np.float32)
        epsilon = 0.001 * cv2.arcLength(points, False)
        smoothed_points = cv2.approxPolyDP(points, epsilon, False)

        total_points = len(smoothed_points)
        indices = np.linspace(0, total_points - 1, self.num_points).astype(int)
        selected_points = smoothed_points[indices]

        x_coords = [point[0][0] for point in selected_points]
        y_coords = [point[0][1] for point in selected_points]

        x_max, x_min = max(x_coords), min(x_coords)
        y_max, y_min = max(y_coords), min(y_coords)
        x_diff, y_diff = x_max - x_min, y_max - y_min

        try:
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = []
                for x, y in zip(x_coords, y_coords):
                    row.extend([f"{x:.2f}", f"{y:.2f}"])
                row.extend([f"{x_max:.2f}", f"{x_min:.2f}", f"{x_diff:.2f}", f"{y_max:.2f}", f"{y_min:.2f}", f"{y_diff:.2f}"])
                writer.writerow(row)
            
            print("已成功將座標和統計數據寫入CSV文件")
            return True
        except IOError:
            print("無法寫入CSV文件")
            return False

    def train_svm_classifier(self, test_size=0.2, kernel='poly', random_state=0):
        #dataset_path = DataSet_Path
        dataset = pd.read_csv(DataSet_Path)
        x = dataset.iloc[:, 1:].values
        y = dataset.iloc[:, 0].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        self.svm_classifier = SVC(kernel=kernel, random_state=random_state)
        self.svm_classifier.fit(x_train, y_train)

        y_pred = self.svm_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test set: {accuracy:.2f}")
    
    def predict_ball_type(self, new_data_path):
        if not self.svm_classifier:
            print("SVM classifier has not been trained.")
            return None

        new_data = pd.read_csv(new_data_path, header=None)
        x_new = new_data.values

        new_pred = self.svm_classifier.predict(x_new)
        ball_types = {0: '直球', 1: '曲球', 2: '滑球'}
        predictions = [ball_types[label] for label in new_pred]
        print("Predictions for new data:", ', '.join(predictions))
        
        return predictions


class Content:
    def __init__(self):
        self.webSocket=WebSocket(self)
    
    def Flow(self):
        asyncio.run(self.webSocket.SocketInit())

class WebSocket:

    def __init__(self,content):
        self.content=content
        self.file_chunks = {}

    async def flow(self, socket, message):
        items = json.loads(message)
        if items['flag'] == 'Upload':
            video_name = (items['filename']).split(".")
            chunk_index = items['chunk']
            total_chunks = items['totalChunks']
            context = items['filebuffer']
            await self.VideoHandler(video_name[0],video_name[1], chunk_index, total_chunks, context,socket)
        elif items['flag']=='Type':
            pass
    
    async def AcceptConnection(self,socket):
        async for message in socket:
           await self.flow(socket=socket,message=message)
          
    async def SendStatus(self,socket,status):
        await socket.send(status)

    async def SocketInit(self):
        async with serve(self.AcceptConnection, IPADDR, PORT,max_size=100*1024*1024):
            await asyncio.Future() 

    async def VideoHandler(self, name,type, chunk_index, total_chunks, context,socket):
        file_data = base64.b64decode(context)
        if name not in self.file_chunks:
            self.file_chunks[name] = [None] * total_chunks
        self.file_chunks[name][chunk_index] = file_data

        if all(chunk is not None for chunk in self.file_chunks[name]):
            video_path = os.path.join(VIDEO_PATH)
            with open(video_path, 'wb') as f:
                for chunk in self.file_chunks[name]:
                    f.write(chunk)
            del self.file_chunks[name]
            await self.SendStatus(socket,json.dumps({"status":True,"detail":"上傳成功","videoPath":f"video"}))
            #self.run_post_processing()
            self.run_posture()
        else:
            print(f"Received chunk {chunk_index + 1}/{total_chunks} for {name}")
    """
    def run_post_processing(self):
        # 執行 subprocess
        subprocess.run([
            "python", r"./detect.py",  # 使用 python 來執行腳本
            "--img", "1920",
            "--conf", "0.7",
            "--device", "0",
            "--weights", "./resource/best.pt",
            "--source", "../baseball/src/Component/video/video.mp4"
        ])

        # 進行後續模型處理
        call_model = Model(image_width=1920, image_height=1080, num_points=10)
        call_model.save_last_frame(VIDEO_PATH, LastFrame_Path)
        call_model.draw_coordinates_on_image(TEXT_PATH, LastFrame_Path, OutPut_Path)
        call_model.process_and_save_coordinates(TEXT_PATH, UnPredicted_Path)
        call_model.train_svm_classifier()
        call_model.predict_ball_type(UnPredicted_Path)
    """
    def run_posture(self):
        subprocess.run(["python",r"process.py"])
    
def main():
    content=Content()
    print(f"server start.Listening on {IPADDR}:{PORT}")
    content.Flow()

if __name__=="__main__":
    main()