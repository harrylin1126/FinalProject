import asyncio
import datetime
import os
import json

from websockets.server import serve

IPADDR="localhost"
PORT=9999

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

    async def flow(self,socket):
        pass
    
    async def AcceptConnection(self,socket):
        async for message in socket:
           print(message)
          
    async def SendStatus(self,socket,status):
        await socket.send(status)

    async def SocketInit(self):
        async with serve(self.AcceptConnection, IPADDR, PORT):
            await asyncio.Future() 

    
def main():
    content=Content()
    print(f"server start.Listening on {IPADDR}:{PORT}")
    content.Flow()
    

if __name__=="__main__":
    main()