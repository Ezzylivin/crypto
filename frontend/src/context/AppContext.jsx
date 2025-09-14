// frontend/src/context/AppContext.jsx

import React, { createContext, useState, useEffect, useContext, useCallback } from 'react';
import api from '../api/api';

const AppContext = createContext();

export function useApp() {
  return useContext(AppContext);
}

export function AppProvider({ children }) {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [isAuthenticated, setIsAuthenticated] = useState(!!token);
  const [logs, setLogs] = useState([]);
  const [isBotRunning, setIsBotRunning] = useState(false);
  const [botStatusLoading, setBotStatusLoading] = useState(true);
  const [settings, setSettings] = useState(null);
  
  const ws = React.useRef(null);

  const connectSocket = useCallback(() => {
    if (!token || (ws.current && ws.current.readyState < 2)) return;

    const wsUrl = (api.defaults.baseURL.replace(/^http/, 'ws')) + `/ws/logs?token=${token}`;
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onopen = () => setLogs(prev => [...prev.slice(-199), "INFO: Log connection established."]);
    ws.current.onmessage = (event) => setLogs(prev => [...prev.slice(-199), event.data]);
    ws.current.onclose = () => setTimeout(connectSocket, 5000);
    ws.current.onerror = () => ws.current.close();
  }, [token]);

  useEffect(() => {
    if (token) {
      localStorage.setItem('token', token);
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      setIsAuthenticated(true);
      connectSocket();
    } else {
      localStorage.removeItem('token');
      delete api.defaults.headers.common['Authorization'];
      setIsAuthenticated(false);
      if (ws.current) ws.current.close();
    }
  }, [token, connectSocket]);

  useEffect(() => {
    if(isAuthenticated) {
      setBotStatusLoading(true);
      api.get("/settings").then(res => {
        setSettings(res.data);
        api.get("/bot/status", { params: { exchange: 'binance', symbol: res.data.default_symbol } })
          .then(res => setIsBotRunning(res.data.running))
          .finally(() => setBotStatusLoading(false));
      }).catch(() => setBotStatusLoading(false));
    }
  }, [isAuthenticated]);

  const login = async (username, password) => {
    const params = new URLSearchParams({ username, password });
    const response = await api.post("/auth/token", params);
    setToken(response.data.access_token);
  };

  const logout = () => setToken(null);

  const startBot = useCallback(async (req) => {
    await api.post("/bot/start", req);
    setIsBotRunning(true);
  }, []);

  const stopBot = useCallback(async (req) => {
    await api.post("/bot/stop", req);
    setIsBotRunning(false);
  }, []);

  const value = {
    isAuthenticated, token, login, logout, logs, setLogs, isBotRunning,
    botStatusLoading, startBot, stopBot, settings, setSettings
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
