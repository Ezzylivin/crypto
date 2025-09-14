// frontend/src/components/Settings.jsx
import React, { useEffect, useState } from "react";
import api from "../api/api";

export default function Settings() {
  const [form, setForm] = useState({
    paper_api_key: "",
    paper_api_secret: "",
    live_api_key: "",
    live_api_secret: "",
    default_symbol: "BTC/USDT",
    max_daily_drawdown_pct: 5.0,
    trading_mode: "paper",
    training_strategy: "static",
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState({ message: "", type: "" });

  useEffect(() => {
    const loadSettings = async () => {
      setLoading(true);
      try {
        const res = await api.get("/settings");
        setForm(res.data);
      } catch (err) {
        setFeedback({ message: "Error loading settings.", type: "error" });
      } finally {
        setLoading(false);
      }
    };
    loadSettings();
  }, []);

  const handleChange = (e) => {
    setFeedback({ message: "", type: "" });
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const saveSettings = async () => {
    setSaving(true);
    setFeedback({ message: "", type: "" });
    try {
      await api.post("/settings", form);
      setFeedback({ message: "Settings saved successfully!", type: "success" });
      setTimeout(() => setFeedback({ message: "", type: "" }), 3000);
    } catch (err) {
      setFeedback({ message: "Error saving settings.", type: "error" });
    } finally {
      setSaving(false);
    }
  };

  const handleModeSwitch = async (newMode) => {
    try {
      const res = await api.post("/settings/trading-mode", { mode: newMode });
      setForm({ ...form, trading_mode: res.data.mode });
      setFeedback({ message: `Trading mode set to ${newMode.toUpperCase()}`, type: "success" });
      setTimeout(() => setFeedback({ message: "", type: "" }), 3000);
    } catch (err) {
      setFeedback({ message: "Failed to switch trading mode.", type: "error" });
    }
  };

  const handleStrategySwitch = async (newStrategy) => {
    try {
      const res = await api.post("/settings/training-strategy", { strategy: newStrategy });
      setForm({ ...form, training_strategy: res.data.strategy });
      setFeedback({ message: `AI Training Strategy set to ${newStrategy.toUpperCase()}`, type: "success" });
      setTimeout(() => setFeedback({ message: "", type: "" }), 3000);
    } catch (err) {
      setFeedback({ message: "Failed to switch training strategy.", type: "error" });
    }
  };

  if (loading) return <p>Loading settings...</p>;

  return (
    <div className="p-6 bg-white rounded-lg shadow-md max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">User Settings</h2>

      <div className="mb-6 p-4 border rounded-lg bg-gray-50">
        <h3 className="font-bold text-lg mb-2 text-gray-700">AI Model Training Strategy</h3>
        <div className="flex items-center space-x-4">
          <button onClick={() => handleStrategySwitch('static')} className={`px-4 py-2 rounded-md font-semibold ${form.training_strategy === 'static' ? 'bg-blue-600 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}>
            Manual (Static)
          </button>
          <button onClick={() => handleStrategySwitch('dynamic')} className={`px-4 py-2 rounded-md font-semibold ${form.training_strategy === 'dynamic' ? 'bg-purple-600 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}>
            Automatic (Dynamic)
          </button>
        </div>
        <p className="text-sm text-gray-600 mt-2">
          {form.training_strategy === 'static' 
             ? "The AI model will only be updated when you manually retrain it."
             : "The AI model will be automatically retrained every 5 days."
          }
        </p>
      </div>

      <div className="mb-6 p-4 border rounded-lg bg-gray-50">
        <h3 className="font-bold text-lg mb-2 text-gray-700">Trading Mode</h3>
        <div className="flex items-center space-x-4">
          <button onClick={() => handleModeSwitch('paper')} className={`px-4 py-2 rounded-md font-semibold ${form.trading_mode === 'paper' ? 'bg-blue-600 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}>
            Paper Trading
          </button>
          <button onClick={() => handleModeSwitch('live')} className={`px-4 py-2 rounded-md font-semibold ${form.trading_mode === 'live' ? 'bg-green-600 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}>
            Live Trading
          </button>
        </div>
      </div>

      <div className="space-y-6">
        <div className="p-4 border-l-4 border-blue-500 bg-blue-50">
          <h3 className="font-semibold text-gray-800 mb-3">Paper Trading Credentials</h3>
          <input name="paper_api_key" type="password" value={form.paper_api_key || ""} onChange={handleChange} placeholder="Paper API Key" className="mt-1 block w-full p-2 border"/>
          <input name="paper_api_secret" type="password" value={form.paper_api_secret || ""} onChange={handleChange} placeholder="Paper API Secret" className="mt-1 block w-full p-2 border"/>
        </div>

        <div className="p-4 border-l-4 border-green-500 bg-green-50">
          <h3 className="font-semibold text-gray-800 mb-3">Live Trading Credentials</h3>
          <input name="live_api_key" type="password" value={form.live_api_key || ""} onChange={handleChange} placeholder="Live API Key" className="mt-1 block w-full p-2 border"/>
          <input name="live_api_secret" type="password" value={form.live_api_secret || ""} onChange={handleChange} placeholder="Live API Secret" className="mt-1 block w-full p-2 border"/>
        </div>

        <div className="p-4 border rounded-lg">
          <h3 className="font-semibold text-gray-700 mb-3">Bot & Risk Defaults</h3>
          <input name="default_symbol" value={form.default_symbol || ""} onChange={handleChange} placeholder="Default Symbol (e.g., BTC/USDT)" className="mt-1 block w-full p-2 border"/>
          <input name="max_daily_drawdown_pct" type="number" value={form.max_daily_drawdown_pct || ""} onChange={handleChange} placeholder="Max Daily Drawdown %" className="mt-1 block w-full p-2 border"/>
        </div>

        {feedback.message && (
          <div className={`p-3 text-center rounded-md text-sm ${feedback.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {feedback.message}
          </div>
        )}

        <button onClick={saveSettings} disabled={saving} className="w-full bg-indigo-600 text-white py-3 rounded-md hover:bg-indigo-700 disabled:bg-gray-400">
          {saving ? "Saving..." : "Save All Settings"}
        </button>
      </div>
    </div>
  );
}
