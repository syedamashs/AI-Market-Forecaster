import './Card.css';

function StockCard(props){

    const isProfit = Number(props.change) >= 0;

    return (
        <div className="card shadow-sm h-100 text-center crypto-card border-0">
            <div className="card-body d-flex flex-column justify-content-center align-items-center">
                <h5 className="card-title">{props.name} {props.symbol ? `(${props.symbol})` : ''}</h5>
                <p className="card-text">Price: {props.price !== null && props.price !== undefined ? `$${props.price}` : '—'}</p>
                {props.change !== null && props.change !== undefined ? (
                    <p style={{color: isProfit? "green":"red", fontWeight: '700'}}>
                        {props.change}%
                    </p>
                ) : null}
            </div>
        </div>
    );
}

export default StockCard;
