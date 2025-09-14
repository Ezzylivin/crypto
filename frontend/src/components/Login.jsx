// frontend/src/components/Login.jsx
import React, { useState } from "react";
import { useApp } from "../context/AppContext";

function Login() {
  const { login } = useApp();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    try {
      await login(username, password);
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed.");
    }
  };

  return (
    <form onSubmit={handleLogin} className="space-y-4 p-8 bg-white rounded-lg shadow-xl w-full max-w-sm">
      <h2 className="text-center text-2xl font-bold text-gray-700">Login</h2>
      {error && <p className="text-center text-red-500 text-sm">{error}</p>}
      <div>
        <input
          type="text" value={username} onChange={(e) => setUsername(e.target.value)}
          placeholder="Username" required
          className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <input
          type="password" value={password} onChange={(e) => setPassword(e.target.value)}
          placeholder="Password" required
          className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition-colors">
        Login
      </button>
    </form>
  );
}

export default Login;
