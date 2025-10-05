import './Card.css';

function CryptoCard(props){

    const isProfit = Number(props.change) >= 0;

    return (
        <div className="card shadow-sm h-100 text-center crypto-card border-0">
            <div className="card-body d-flex flex-column justify-content-center align-items-center">
                <img src={props.image} alt={props.name} className="mb-2" style={{width:"40px" , height:"40px"}} />
                <h5 className="card-title">{props.name}</h5>
                <p className="card-text">Price: ${props.price}</p>
                <p style={{color: isProfit? "green":"red", fontWeight: '700'}}>
                    {props.change}%
                </p>
            </div>
        </div>
    );
}

export default CryptoCard;