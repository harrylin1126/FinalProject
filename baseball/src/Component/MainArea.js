import React from "react";
import UploadButton from "./SelectField";
import NavBar from "./DivBar";
import Button from "./Button";
import ResultArea from "./ResultArea";
import { ws } from './WebServer'


export default class MainArea extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            isUpload:true,
            videoPath:undefined
        }
    }

    UploadHandler(path)
    {
        this.setState({
            isUpload:true,
            videoPath:path
        })
    }

    BallTypeHandler()
    {
        ws.send(JSON.stringify({
            flag: "Type",
        }));
        ws.onmessage = (result) => {
            let msg = JSON.parse(result.data)
            alert(msg['detail'])
        }
    }

    PostHandler()
    {
        ws.send(JSON.stringify({
            flag: "Post",
        }));
        ws.onmessage = (result) => {
            let msg = JSON.parse(result.data)
            alert(msg['detail'])
        }
        
    }

    render() {
        if(this.state.isUpload)
            return(
                <div>
                <NavBar></NavBar>
                <ResultArea state={this.state.isUpload} videoPath={this.state.videoPath}></ResultArea>
                <div style={{
                    display:"flex",
                    justifyContent:'space-between',
                    alignContent:'center'
                }}>
                <Button description="球種" action={()=>this.BallTypeHandler()}></Button>
                <Button description="姿勢" action={()=>this.PostHandler()}></Button>
                </div>
            </div>
        )
        else
            return (
                <div>
                    <NavBar></NavBar>
                    <UploadButton uploadHandler={(e)=>this.UploadHandler(e)}></UploadButton>
                </div>
        )
    }

}

