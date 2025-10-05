import './Card.css';

const stockLogos = {
  AAPL: "https://logo.clearbit.com/apple.com",
  MSFT: "https://logo.clearbit.com/microsoft.com",
  GOOGL: "https://logo.clearbit.com/abc.xyz",
  AMZN: "https://logo.clearbit.com/amazon.com",
  TSLA: "https://logo.clearbit.com/tesla.com",
  NVDA: "https://logo.clearbit.com/nvidia.com",
  META: "https://logo.clearbit.com/meta.com",
  JPM: "https://logo.clearbit.com/jpmorganchase.com",
  KO: "https://logo.clearbit.com/coca-cola.com",
  NKE: "https://logo.clearbit.com/nike.com",
  DIS: "https://logo.clearbit.com/disney.com",
  WMT: "https://logo.clearbit.com/walmart.com",
  AMC: "https://logo.clearbit.com/amc.com",
  GME: "https://logo.clearbit.com/gamestop.com",
};

function StockCard(props) {
  const isProfit = Number(props.change) >= 0;

  const logo = stockLogos[props.symbol] || "https://via.placeholder.com/50?text=Stock";

  return (
    <div 
    className="card shadow-sm h-100 text-center stock-card border-0"
    onClick={props.onClick}      
    style={{ cursor: 'pointer' }} 
    >
      <div className="card-body d-flex flex-column justify-content-center align-items-center">
        <img
          src={logo}
          alt={props.name}
          style={{ width: '50px', height: '50px', marginBottom: '10px', borderRadius: '5px' }}
        />
        <h5 className="card-title">{props.name} ({props.symbol})</h5>
        <p className="card-text">Price: {props.price !== null ? `$${props.price.toFixed(2)}` : '—'}</p>
        {props.change !== null && (
          <p style={{ color: isProfit ? 'green' : 'red', fontWeight: '700' }}>
            {props.change}%
          </p>
        )}
      </div>
    </div>
  );
}

export default StockCard;
