import React from "react";
import UploadButton from "./SelectField";
import NavBar from "./DivBar";
import Button from "./Button";
import ResultArea from "./ResultArea";


export default class MainArea extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            isUpload:false,
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
                <Button description="球種"></Button>
                <Button description="姿勢"></Button>
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

