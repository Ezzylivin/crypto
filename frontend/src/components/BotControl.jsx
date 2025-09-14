// frontend/src/components/BotControl.jsx
import React, { useState } from "react";
import { useApp } from "../context/AppContext";

export default function BotControl() {
  const { isBotRunning, botStatusLoading, startBot, stopBot, settings } = useApp();

  const [exchange, setExchange] = useState("binance");
  const [symbol, setSymbol] = useState(settings?.default_symbol || "BTC/USDT");
  const [strategy, setStrategy] = useState(settings?.default_strategy || "default");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleStart = async () => {
    setError("");
    setLoading(true);
    try {
      await startBot({ exchange, symbol, strategy });
    } catch (err) {
      setError(err.response?.data?.detail || "Error starting bot.");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setError("");
    setLoading(true);
    try {
      await stopBot({ exchange, symbol, strategy });
    } catch (err) {
      setError(err.response?.data?.detail || "Error stopping bot.");
    } finally {
      setLoading(false);
    }
  };

  const isActionDisabled = loading || botStatusLoading;

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Bot Control</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <input value={exchange} onChange={e => setExchange(e.target.value)} className="border p-2 rounded-md" />
        <input value={symbol} onChange={e => setSymbol(e.target.value)} className="border p-2 rounded-md" />
        <input value={strategy} onChange={e => setStrategy(e.target.value)} className="border p-2 rounded-md" />
      </div>
      <div className="flex items-center gap-4">
        <button onClick={handleStart} disabled={isBotRunning || isActionDisabled} className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:bg-gray-400">Start Bot</button>
        <button onClick={handleStop} disabled={!isBotRunning || isActionDisabled} className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 disabled:bg-gray-400">Stop Bot</button>
        <div className={`text-lg font-semibold p-2 rounded-md ${isBotRunning ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          Status: {botStatusLoading ? "Loading..." : (isBotRunning ? "Running" : "Stopped")}
        </div>
      </div>
      {error && <p className="mt-4 text-red-600">{error}</p>}
    </div>
  );
}
