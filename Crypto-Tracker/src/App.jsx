import { useEffect , useState } from "react";
import CryptoCard from "./components/CryptoCard";
import StockCard from "./components/StockCard";
import './App.css';

function App(){

  const [coins , setCoins] = useState([]);
  const [stocks, setStocks] = useState([]);
  const [market, setMarket] = useState('crypto'); // 'crypto' or 'stocks'
  const [search , setSearch] = useState("");

  useEffect(() => {
    // load coins once
    fetch("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd")
    .then(res => res.json())
    .then(data => setCoins(data))
    .catch(err => console.error("Error fetching coins!!",err));
  }, []);

  // stocks will be fetched on demand when user searches in 'stocks' market

  const filteredcoins = coins.filter((coin) => 
    coin.name.toLowerCase().includes(search.toLowerCase())
  );

  const openPredictor = () => {
    // open Streamlit app in a new tab — ensure Streamlit is running locally on port 8501
    window.open('http://localhost:8501', '_blank');
  }

  const toggleMarket = (m) => {
    setMarket(m);
    setSearch('');
    setStocks([]);
  }

  // perform stock search via Yahoo Finance (public endpoints)
  const searchStocks = async (query) => {
    if(!query || !query.trim()){
      setStocks([]);
      return;
    }
    try{
      // quick search endpoint
      const searchRes = await fetch(`https://query1.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(query)}`);
      const sjson = await searchRes.json();
      const quotes = (sjson.quotes || []).slice(0, 20);
      // for each symbol, fetch quote summary to get price and change
      const mapped = await Promise.all(quotes.map(async (q) => {
        const symbol = q.symbol;
        try{
          const qres = await fetch(`https://query1.finance.yahoo.com/v7/finance/quote?symbols=${encodeURIComponent(symbol)}`);
          const qj = await qres.json();
          const r = qj.quoteResponse.result[0];
          return {
            symbol: symbol,
            name: r.longName || r.shortName || q.shortname || symbol,
            price: r.regularMarketPrice,
            change: r.regularMarketChangePercent ? r.regularMarketChangePercent.toFixed(2) : null
          }
        }catch(e){
          return { symbol, name: q.shortname || q.longname || symbol, price: null, change: null };
        }
      }));
      setStocks(mapped);
    }catch(err){
      console.error('Error searching stocks', err);
      setStocks([]);
    }
  }

  const isSearching = search.trim().length > 0;

  return (
    <div className="app-root">
      <header className="app-header">
        <button className="predictor-btn" onClick={openPredictor} aria-label="Open Predictor">Predictor</button>
      </header>

      <main className="app-main">
        <div className={`center-box ${isSearching ? 'top' : 'centered'}`}>
          <div className="left-toggle">
            <button className={`market-btn ${market==='crypto' ? 'active' : ''}`} onClick={() => toggleMarket('crypto')}>Crypto</button>
            <button className={`market-btn ${market==='stocks' ? 'active' : ''}`} onClick={() => toggleMarket('stocks')}>Stocks</button>
          </div>
          <h1 className="app-title">{market === 'crypto' ? 'Crypto-Tracker' : 'Stock-Tracker'}</h1><br />
          <div className={`search-container`}>
            <input 
             type="text"
             placeholder={market === 'crypto' ? "Search coins (e.g. Bitcoin, Ethereum)" : "Search stocks (e.g. AAPL, Microsoft)"}
             value={search}
             onChange={(e) => {
               const v = e.target.value;
               setSearch(v);
               if(market === 'stocks'){
                 // debounce not implemented for brevity; simple immediate search
                 searchStocks(v);
               }
             }}
             className={`search-input ${isSearching ? 'small' : 'large'}`}
            />
          </div>
          {!isSearching && (
            <p className="hint">Type a name above and press Enter or click to search.</p>
          )}
        </div>

        {/* only show results after the user has typed something */}
        {search.trim().length > 0 ? (
          <div className="results d-flex flex-wrap justify-content-center gap-4">
            {market === 'crypto' ? (
              filteredcoins.length ? filteredcoins.map((coin) => (
                <div key={coin.id} className="p-2" style={{width: "250px"}}>
                  <CryptoCard 
                    key={coin.id}
                    name={coin.name}
                    price={coin.current_price}
                    image={coin.image}
                    change={coin.price_change_percentage_24h ? coin.price_change_percentage_24h.toFixed(2) : 0}
                  />
                </div>
              )) : (
                <p className="no-results">No coins found for "{search}"</p>
              )
            ) : (
              stocks.length ? stocks.map((s) => (
                <div key={s.symbol} className="p-2" style={{width: "250px"}}>
                  <StockCard 
                    symbol={s.symbol}
                    name={s.name}
                    price={s.price}
                    change={s.change}
                  />
                </div>
              )) : (
                <p className="no-results">No stocks found for "{search}"</p>
              )
            )}
          </div>
        ) : ( null
        )}
      </main>
    </div>
  );
}

export default App;