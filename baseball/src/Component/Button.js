export default function Button({action,description})
{   
    
    return(
        <div style={{
            width:'100%',
            alignContent:'center'

        }}>
            <button style={{
                backgroundColor:'rgba(73,107,191,1.0)',
                width:'200px',
                margin:"30px auto",
                color:'white',
                border:'none',
                borderRadius:"10px"
                }} className="btn btn-light"
                 onClick={()=>action()}>{description}</button>
        </div>
    )
}