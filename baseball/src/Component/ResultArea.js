import { useEffect, useState } from "react";
import {ws} from "./WebServer"
import video from './video/video.mp4';

export default function ResultArea({state,videoPath})
{
    const [imgState,setimgState]=useState(state)
    const [img,setImg]=useState([])

    useEffect(()=>{
        ws.onmessege=(e)=>{
            setimgState(true)
        }
    })

    useEffect(()=>{
        if(imgState)
            setImg(VideoFactor(videoPath))
    },[imgState])

    return(
        <div style={{
            height:'50vh',
            width:'80%',
            margin:"30px auto",
            backgroundColor:'rgba(73,107,191,1.0)',
        }}>
            {img}
        </div>
    )
}

function VideoFactor(path)
{
    return (<video width="100%" height="100%" autoPlay muted>
        <source src={video} type="video/mp4"/>
    </video>)
}