import React, { useState } from "react";
import { ws } from './WebServer'

export default function UploadButton({uploadHandler }) {

    const CHUNK_SIZE = 1024 * 1024;
    const [selectedFiles, setSelectedFiles] = useState(null);
    ws.onmessage = (e) => {
        let msg = JSON.parse(e.data)
        uploadHandler(msg['videoPath'])
        alert(msg['detail'])

    }


    const handleFileChange = (event) => {
        setSelectedFiles(event.target.files);
    };

    const handleUpload = () => {
        if (ws && selectedFiles) {
            for (let i = 0; i < selectedFiles.length; i++) {
                const file = selectedFiles[i];
                const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
                let currentChunk = 0;

                const reader = new FileReader();

                reader.onload = function (e) {
                    const fileBuffer = e.target.result;
                    try{

                        ws.send(JSON.stringify({
                            flag: "Upload",
                            filename: file.name,
                            chunk: currentChunk,
                            totalChunks: totalChunks,
                            filebuffer: btoa(new Uint8Array(fileBuffer).reduce((data, byte) => data + String.fromCharCode(byte), ''))
                        }));
                    }
                    catch(e)
                    {
                        console.error(e)
                    }

                    currentChunk++;
                    if (currentChunk < totalChunks) {
                        sendNextChunk();  // 傳送下一個塊
                    } else {
                        console.log(`${file.name} 上傳完成`);
                    }
                };

                function sendNextChunk() {
                    const start = currentChunk * CHUNK_SIZE;
                    const end = Math.min(start + CHUNK_SIZE, file.size);
                    const chunk = file.slice(start, end);
                    reader.readAsArrayBuffer(chunk);  // 讀取每個塊為 ArrayBuffer
                }

                sendNextChunk();  // 開始傳送第一個塊
            }
        } else {
            alert("沒有選擇檔案");
        }
    };

    return (
        <div style={{
            width: '80%',
            margin: 'auto'
        }}>
            <div className="mb-3">
                <h1 for="formFileMultiple" className="form-label">選擇影片</h1>
                <input className="form-control" type="file" id="formFileMultiple" multiple onChange={(e) => handleFileChange(e)} />
                <button style={{ margin: '30px', color: 'white', backgroundColor: 'rgba(73,107,191,1.0)', }} type="button" className="btn btn-light" onClick={() => handleUpload()}>上傳</button>
            </div>
            <div>
                {/* <h2 for="formFileMultiple" className="form-label">已選擇資料</h2>
            <ul style={{listStyle:"none"}}>
                
                {reactChildren}
            </ul> */}
            </div>
        </div>
    )
}