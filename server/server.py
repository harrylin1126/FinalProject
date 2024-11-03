import asyncio
import datetime
import os
import json
import base64

from websockets.server import serve

IPADDR="localhost"
PORT=9999
VIDEO_PATH="../baseball/src/Component/video"

class model:
    pass

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
            video_path = os.path.join(VIDEO_PATH, 'video.'+type)
            with open(video_path, 'wb') as f:
                for chunk in self.file_chunks[name]:
                    f.write(chunk)
            del self.file_chunks[name]
            await self.SendStatus(socket,json.dumps({"status":True,"detail":"上傳成功","videoPath":f"video"}))
        else:
            print(f"Received chunk {chunk_index + 1}/{total_chunks} for {name}")

    
def main():
    content=Content()
    print(f"server start.Listening on {IPADDR}:{PORT}")
    content.Flow()
    

if __name__=="__main__":
    main()