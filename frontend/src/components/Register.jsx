// frontend/src/components/Register.jsx
import React, { useState } from "react";
import api from "../api/api";
import { Link } from 'react-router-dom';

function Register() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (password.length < 4) {
      setError("Password must be at least 4 characters long.");
      return;
    }

    try {
      const response = await api.post("/auth/register", { username, password });
      setSuccess(response.data.message);
    } catch (err) {
      setError(err.response?.data?.detail || "Registration failed.");
    }
  };

  return (
    <div className="space-y-4 p-8 bg-white rounded-lg shadow-xl w-full max-w-sm">
      <form onSubmit={handleRegister} className="space-y-4">
        <h2 className="text-center text-2xl font-bold text-gray-700">Register New Account</h2>
        {error && <p className="text-center text-red-500 text-sm">{error}</p>}
        {success && <p className="text-center text-green-500 text-sm">{success}</p>}
        <div>
          <input
            type="text" value={username} onChange={(e) => setUsername(e.target.value)}
            placeholder="Username" required
            className="w-full px-4 py-2 border border-gray-300 rounded-md"
          />
        </div>
        <div>
          <input
            type="password" value={password} onChange={(e) => setPassword(e.target.value)}
            placeholder="Password" required
            className="w-full px-4 py-2 border border-gray-300 rounded-md"
          />
        </div>
        <button type="submit" className="w-full bg-green-600 text-white py-2 rounded-md hover:bg-green-700">
          Register
        </button>
      </form>
      <div className="text-center">
        <Link to="/login" className="text-sm text-blue-600 hover:underline">
          Already have an account? Login
        </Link>
      </div>
    </div>
  );
}

export default Register;
