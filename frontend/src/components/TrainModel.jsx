// frontend/src/components/TrainModel.jsx 
import React, { useState } from "react"; import api from "../api/api";
export default function TrainModel() { const [status, setStatus] = useState(""); const [loading, setLoading] = useState(false);
const train = async () => { setLoading(true); setStatus("Requesting model training..."); try { const response = await api.post("/train-model"); setStatus(response.data.message); } catch (err) { setStatus(err.response?.data?.detail || "Error requesting training."); } finally { setLoading(false); } };
return ( <div className="p-6 bg-white rounded-lg shadow-md"> <h2 className="text-2xl font-bold mb-4 text-gray-800">Train AI Models</h2> <p className="mb-4 text-gray-600"> Start the ensemble model training process. This uses your downloaded data to create all necessary models (regime, trending, and ranging). Check logs for progress. </p> <button onClick={train} disabled={loading} className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 disabled:bg-gray-400"> {loading ? "Training in Progress..." : "Start Training"} </button> {status && <p className="mt-4 text-gray-700 font-semibold">{status}</p>} </div> ); }

