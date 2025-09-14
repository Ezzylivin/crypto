// frontend/src/components/Downloader.jsx
import React, { useState } from "react";
import api from "../api/api";
import { useApp } from '../context/AppContext';

export default function Downloader() {
  const { settings } = useApp();
  const [exchange, setExchange] = useState("binance");
  const [symbol, setSymbol] = useState(settings?.default_symbol || "BTC/USDT");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const download = async () => {
    setLoading(true);
    setStatus("Requesting download...");
    try {
      const response = await api.post("/download-data", { exchange, symbol });
      setStatus(response.data.message || "Download started. Check logs for progress.");
    } catch (err) {
      setStatus(err.response?.data?.detail || "Error requesting download.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Download Market Data</h2>
      <p className="mb-4 text-gray-600">Download historical OHLCV data to be used in training and backtesting.</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <input value={exchange} onChange={e => setExchange(e.target.value)} placeholder="Exchange" className="border p-2 rounded-md" />
        <input value={symbol} onChange={e => setSymbol(e.target.value)} placeholder="Symbol (e.g., BTC/USDT)" className="border p-2 rounded-md" />
      </div>
      <button onClick={download} disabled={loading} className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400">
        {loading ? "Downloading..." : "Download Data"}
      </button>
      {status && <p className="mt-4 text-gray-700 font-semibold">{status}</p>}
    </div>
  );
}
