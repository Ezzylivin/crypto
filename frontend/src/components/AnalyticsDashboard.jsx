// frontend/src/components/AnalyticsDashboard.jsx
import React, { useState, useEffect } from 'react';
import api from '../api/api';

export default function AnalyticsDashboard() {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchTradeHistory = async () => {
      try {
        setLoading(true);
        const response = await api.get('/analytics/trade-history');
        setTrades(response.data);
      } catch (err) {
        setError('Failed to load trade history.');
      } finally {
        setLoading(false);
      }
    };
    fetchTradeHistory();
  }, []);

  const formatTimestamp = (ts) => ts ? new Date(ts).toLocaleString() : 'N/A';
  const formatPnl = (pnl) => {
    if (pnl === null || pnl === undefined) return 'N/A';
    const isProfit = pnl >= 0;
    const color = isProfit ? 'text-green-500' : 'text-red-500';
    const sign = isProfit ? '+' : '';
    return <span className={color}>{`${sign}${pnl.toFixed(2)}%`}</span>;
  };

  if (loading) return <p className="text-center">Loading trade history...</p>;
  if (error) return <p className="text-center text-red-500">{error}</p>;

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Trade History</h2>
      {trades.length === 0 ? (
        <p className="text-gray-500">No trades have been recorded yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Entry Reason</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Entry</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Exit</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">PnL</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {trades.map((trade) => (
                <tr key={trade.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{trade.symbol}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{trade.entry_reason}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <div>${trade.entry_price.toFixed(4)}</div>
                    <div className="text-xs text-gray-400">{formatTimestamp(trade.entry_timestamp)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <div>{trade.exit_price ? `$${trade.exit_price.toFixed(4)}` : 'In Position'}</div>
                    <div className="text-xs text-gray-400">{formatTimestamp(trade.exit_timestamp)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold">{formatPnl(trade.profit_loss_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}```

#### **`frontend/src/components/BacktestChart.jsx`**
```jsx
// frontend/src/components/BacktestChart.jsx
import React, { useState } from "react";
import { Line } from "react-chartjs-2";
import { Chart, LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend } from "chart.js";
import api from "../api/api";

Chart.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

export default function BacktestChart() {
  const [chartData, setChartData] = useState(null);
  const [stats, setStats] = useState(null);
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [exchange, setExchange] = useState("binance");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchData = async () => {
    setLoading(true);
    setError("");
    setChartData(null);
    setStats(null);
    try {
      const res = await api.post("/backtest", { exchange, symbol });
      setStats({
        total_return: res.data.total_return,
        trades: res.data.trades,
        win_rate: res.data.win_rate,
      });
      setChartData({
        labels: res.data.timestamps,
        datasets: [{
          label: "PnL Curve (%)",
          data: res.data.pnl_curve,
          borderColor: "rgb(75, 192, 192)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          fill: true,
          tension: 0.1,
        }],
      });
    } catch (err) {
      setError(err.response?.data?.detail || "Error fetching backtest data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Backtest PnL Chart</h2>
      <div className="flex items-center gap-4 mb-4">
        <input value={exchange} onChange={(e) => setExchange(e.target.value)} placeholder="Exchange" className="border p-2 rounded-md" />
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol" className="border p-2 rounded-md" />
        <button onClick={fetchData} className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700" disabled={loading}>
          {loading ? "Running Backtest..." : "Run Backtest"}
        </button>
      </div>

      {error && <p className="text-red-600">{error}</p>}

      {stats && (
        <div className="grid grid-cols-3 gap-4 text-center mb-4 p-4 bg-gray-100 rounded-md">
          <div>
            <div className="text-sm text-gray-500">Total Return</div>
            <div className={`text-2xl font-bold ${stats.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {stats.total_return.toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Total Trades</div>
            <div className="text-2xl font-bold text-gray-800">{stats.trades}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Win Rate</div>
            <div className="text-2xl font-bold text-gray-800">{stats.win_rate.toFixed(2)}%</div>
          </div>
        </div>
      )}
      
      {chartData ? (
        <Line data={chartData} options={{ responsive: true }} />
      ) : (
        !loading && <p className="text-center text-gray-500">Run a backtest to see the results chart.</p>
      )}
    </div>
  );
}
