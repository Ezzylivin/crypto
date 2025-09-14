// frontend/src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, Link } from 'react-router-dom';
import Login from './components/Login';
import Register from './components/Register';
import BotControl from './components/BotControl';
import Logs from './components/Logs';
import Settings from './components/Settings';
import Downloader from './components/Downloader';
import TrainModel from './components/TrainModel';
import ModelManager from './components/ModelManager';
import BacktestChart from './components/BacktestChart';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import { AppProvider, useApp } from './context/AppContext';

function MainApp() {
  const { isAuthenticated, logout } = useApp();

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="*" element={<Navigate to="/login" />} />
        </Routes>
      </div>
    );
  }

  return (
     <div className="min-h-screen bg-gray-100">
        <header className="bg-white shadow">
          <div className="container mx-auto px-4 py-4 flex justify-between items-center">
             <h1 className="text-xl font-bold text-gray-800">AI Trading Dashboard</h1>
             <button onClick={logout} className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded">
                Logout
             </button>
          </div>
        </header>
        <nav className="bg-gray-800 text-white">
          <div className="container mx-auto px-1 flex flex-wrap space-x-1">
            <Link to="/bot" className="px-3 py-3 hover:bg-gray-700">Bot Control</Link>
            <Link to="/analytics" className="px-3 py-3 hover:bg-gray-700">Analytics</Link>
            <Link to="/logs" className="px-3 py-3 hover:bg-gray-700">Live Logs</Link>
            <Link to="/settings" className="px-3 py-3 hover:bg-gray-700">Settings</Link>
            <Link to="/data" className="px-3 py-3 hover:bg-gray-700">Data Manager</Link>
            <Link to="/training" className="px-3 py-3 hover:bg-gray-700">Training</Link>
            <Link to="/backtest" className="px-3 py-3 hover:bg-gray-700">Backtest</Link>
          </div>
        </nav>
        <main className="container mx-auto p-4">
           <Routes>
              <Route path="/bot" element={<BotControl />} />
              <Route path="/analytics" element={<AnalyticsDashboard />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/data" element={<div className="space-y-4"><Downloader /><ModelManager/></div>} />
              <Route path="/training" element={<TrainModel />} />
              <Route path="/backtest" element={<BacktestChart />} />
              <Route path="/" element={<Navigate to="/bot" />} />
           </Routes>
        </main>
      </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppProvider>
        <MainApp />
      </AppProvider>
    </BrowserRouter>
  );
}

export default App;
