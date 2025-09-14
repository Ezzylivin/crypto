// frontend/src/components/Logs.jsx
import React, { useEffect, useRef } from "react";
import { useApp } from "../context/AppContext";

export default function Logs() {
  const { logs, setLogs } = useApp();
  const logContainerRef = useRef(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">Live Logs</h2>
        <button onClick={() => setLogs([])} className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600">Clear</button>
      </div>
      <div ref={logContainerRef} className="h-96 overflow-y-auto bg-black text-green-400 font-mono p-4 rounded-md text-sm">
        {logs.map((line, i) => (
          <div key={i} className={line.includes("ERROR") || line.includes("FATAL") || line.includes("CRITICAL") ? "text-red-400" : ""}>{`> ${line}`}</div>
        ))}
      </div>
    </div>
  );
}
