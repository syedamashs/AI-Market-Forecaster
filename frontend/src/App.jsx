import { useEffect, useState } from "react";
import CryptoCard from "./components/CryptoCard";
import StockCard from "./components/StockCard";
import './App.css';

function App() {
  const [coins, setCoins] = useState([]);
  const [stocks, setStocks] = useState([]);
  const [market, setMarket] = useState('crypto'); 
  const [search, setSearch] = useState("");
  const [selectedItem, setSelectedItem] = useState(null); 

  const apiKey = "d3h7depr01qstnq80tm0d3h7depr01qstnq80tmg";

  useEffect(() => {
    fetch("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd")
      .then(res => res.json())
      .then(data => setCoins(data))
      .catch(err => console.error("Error fetching coins!!", err));
  }, []);

  const filteredCoins = coins.filter((coin) =>
    coin.name.toLowerCase().startsWith(search.toLowerCase())
  );

  const openPredictor = () => {
    window.open('http://localhost:8501', '_blank');
  };

  const toggleMarket = (m) => {
    setMarket(m);
    setSearch('');
    setStocks([]);
  };

  const searchStocks = async (query) => {
    if (!query || !query.trim()) {
      setStocks([]);
      return;
    }

    const q = query.trim().toUpperCase();

    try {
      const searchRes = await fetch(`https://finnhub.io/api/v1/search?q=${q}&token=${apiKey}`);
      const searchData = await searchRes.json();
      const results = searchData.result || [];
      if (results.length === 0) {
        setStocks([]);
        return;
      }
      const topResults = results.slice(0, 20);

      const mapped = await Promise.all(topResults.map(async (item) => {
        const symbol = item.symbol;
        try {
          const quoteRes = await fetch(`https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${apiKey}`);
          const quote = await quoteRes.json();
          if (quote && quote.c) {
            return {
              symbol,
              name: item.description || symbol,
              price: quote.c,
              change: ((quote.c - quote.pc) / quote.pc * 100).toFixed(2),
              high: quote.h,
              low: quote.l,
              open: quote.o,
              previousClose: quote.pc
            };
          } else {
            return { symbol, name: item.description || symbol, price: null, change: null };
          }
        } catch {
          return { symbol, name: item.description || symbol, price: null, change: null };
        }
      }));

      setStocks(mapped);
    } catch (err) {
      console.error("Error fetching stocks:", err);
      setStocks([]);
    }
  };

  const isSearching = search.trim().length > 0;

  return (
    <div className="app-root">

      <header className="app-header">
        <div className="market-toggle">
          <input type="checkbox" id="marketSwitch" checked={market === 'stocks'} onChange={() => toggleMarket(market === 'crypto' ? 'stocks' : 'crypto')} />
        <label htmlFor="marketSwitch">
          <span className="toggle-left">Crypto</span>
          <span className="toggle-right">Stocks</span>
          <span className="toggle-slider"></span>
        </label>
        </div>
      <button className="predictor-btn" onClick={openPredictor} aria-label="Open Predictor">Predictor</button>
    </header> 


      <main className="app-main">
        <div className={`center-box ${isSearching ? 'top' : 'centered'}`}>
          <h1 className="app-title">{market === 'crypto' ? 'Crypto-Tracker' : 'Stock-Tracker'}</h1><br />

          <div className={`search-container`}>
            <input type="text" placeholder={market === 'crypto' ? "Search coins (e.g. Bitcoin, Ethereum)" : "Search stocks (e.g. MSFT, AAPL)"} value={search} onChange={(e) => {
                const v = e.target.value;
                setSearch(v);
                if (market === 'stocks') searchStocks(v);
              }}
              className={`search-input ${isSearching ? 'small' : 'large'}`}
            />
          </div>

          {!isSearching && (
            <p className="hint">Type a name above and press Enter or click to search.</p>
          )}
        </div>

        {isSearching && (
          <div className="results d-flex flex-wrap justify-content-center gap-4">
            {market === 'crypto' ? (
              filteredCoins.length ? filteredCoins.map((coin) => (
                <div key={coin.id} className="p-2" style={{ width: "250px" }}>
                  <CryptoCard
                    name={coin.name}
                    price={coin.current_price}
                    image={coin.image}
                    change={coin.price_change_percentage_24h?.toFixed(2) || 0}
                    onClick={() => setSelectedItem({ type: 'crypto', data: coin })}
                  />
                </div>
              )) : (
                <p className="no-results">No coins found for "{search}"</p>
              )
            ) : (
              stocks.length ? stocks.map((s) => (
                <div key={s.symbol} className="p-2" style={{ width: "250px" }}>
                  <StockCard
                    symbol={s.symbol}
                    name={s.name}
                    price={s.price}
                    change={s.change}
                    onClick={() => setSelectedItem({ type: 'stock', data: s })}
                  />
                </div>
              )) : (
                <p className="no-results">No stocks found for "{search}"</p>
              )
            )}
          </div>
        )}
      </main>

      {/* Modal */}
      {selectedItem && (
        <div className="modal-overlay" onClick={() => setSelectedItem(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setSelectedItem(null)}>X</button>

            {selectedItem.type === 'crypto' ? (
              <>
                <img src={selectedItem.data.image} alt={selectedItem.data.name} style={{ width: '70px', marginBottom: '10px' }} />
                <h2>{selectedItem.data.name}</h2>
                <p>Price: ${selectedItem.data.current_price}</p>
                <p>Market Cap: ${selectedItem.data.market_cap.toLocaleString()}</p>
                <p>24h High: ${selectedItem.data.high_24h}</p>
                <p>24h Low: ${selectedItem.data.low_24h}</p>
                <p>24h Change: {selectedItem.data.price_change_percentage_24h?.toFixed(2)}%</p>
              </>
            ) : (
              <>
                <h2>{selectedItem.data.name} ({selectedItem.data.symbol})</h2>
                <p>Price: ${selectedItem.data.price}</p>
                <p>Change: {selectedItem.data.change}%</p>
                <p>Open: ${selectedItem.data.open}</p>
                <p>High: ${selectedItem.data.high}</p>
                <p>Low: ${selectedItem.data.low}</p>
                <p>Previous Close: ${selectedItem.data.previousClose}</p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
