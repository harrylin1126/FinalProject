import React from "react";

export default class NavBar extends React.Component
{
    constructor(props)
    {
        super(props)
        this.state={

        }
    }

    render()
    {
        return (
            <div style={{
                height:'10vh',
                width:"100%",
                backgroundColor:'rgba(73,107,191,1.0)',
                color:'white',
            }}>
                <nav>
                    <h1 style={{
                    width:'50%',
                    height:'100%',
                    textAlign:'center',
                    margin:'auto',
                    alignContent:'center',
                    fontSize:"75px"
                }}>
                        KANO辨球
                    </h1>
                </nav>
            </div>

        )
    }
}