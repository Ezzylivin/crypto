// frontend/src/components/ModelManager.jsx
import React, { useEffect, useState } from "react";
import api from "../api/api";

export default function ModelManager() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadModels = async () => {
    try {
      const res = await api.get("/models");
      setModels(res.data); 
    } catch (err) {
      console.error("Error loading models", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const deleteModel = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) return;
    try {
      await api.delete(`/models/${encodeURIComponent(filename)}`);
      loadModels();
    } catch (err) {
      alert(err.response?.data?.detail || "Error deleting model.");
    }
  };

  if (loading) return <p>Loading models...</p>;

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Model Manager</h2>
      {models.length === 0 ? (
        <p className="text-gray-500">No trained models found. Go to the "Training" tab to create them.</p>
      ) : (
        <ul className="space-y-2">
          {models.map((name) => (
            <li key={name} className="flex justify-between items-center border p-3 rounded-md bg-gray-50">
              <span className="font-mono text-gray-700">{name}</span>
              <button
                onClick={() => deleteModel(name)}
                className="bg-red-600 text-white px-3 py-1 rounded-md text-sm hover:bg-red-700"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
