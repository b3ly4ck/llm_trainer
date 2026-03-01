import React, { useState, useEffect } from 'react';
import LossChart from './components/LossChart';

export default function App() {
    const [file, setFile] = useState<File | null>(null);
    const [datasetPath, setDatasetPath] = useState('');
    const [method, setMethod] = useState('lora');
    const [baseModel, setBaseModel] = useState('meta-llama/Llama-2-7b-hf');
    const [taskId, setTaskId] = useState<number | null>(null);
    const [status, setStatus] = useState('Ожидание');
    const [metrics, setMetrics] = useState<Array<{ step: number; loss: number }>>([]);
    const [error, setError] = useState<string | null>(null);

    const handleUpload = async () => {
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const res = await fetch('/api/dataset/upload', { 
                method: 'POST', 
                body: formData 
            });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Ошибка загрузки');
            
            setDatasetPath(data.file_path);
            alert('Файл успешно загружен и прошел проверку!');
            setError(null);
        } catch (err: any) {
            setError(err.message);
        }
    };

    const startTraining = async () => {
        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_path: datasetPath,
                    params: { 
                        method: method, 
                        model_name: baseModel, 
                        epochs: 3, 
                        lr: method === 'full' ? 0.00001 : 0.0002 
                    }
                })
            });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Ошибка запуска');
            
            setTaskId(data.task_id);
            setStatus('Инициализация...');
            setMetrics([]);
            setError(null);
        } catch (err: any) {
            setError(err.message);
        }
    };

    useEffect(() => {
        if (!taskId) return;
        
        // Подключаемся к вебсокету по текущему хосту, чтобы работало и локально, и на сервере
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/metrics/${taskId}`);
        
        ws.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.status) setStatus(data.status);
            if (data.error) setError(data.error);
            if (data.loss !== undefined && data.step !== undefined) {
                setMetrics(prev => [...prev, { step: data.step, loss: data.loss }]);
            }
        };
        
        return () => ws.close();
    }, [taskId]);

    return (
        <div style={{ padding: '40px', maxWidth: '900px', margin: '0 auto', fontFamily: 'sans-serif', backgroundColor: '#f9f9f9', minHeight: '100vh' }}>
            <h1 style={{ textAlign: 'center', marginBottom: '40px' }}>LLM Fine-Tuning Dashboard</h1>
            
            {error && (
                <div style={{ padding: '15px', backgroundColor: '#ffebee', color: '#c62828', marginBottom: '20px', borderRadius: '5px' }}>
                    <strong>Ошибка: </strong> {error}
                </div>
            )}

            <section style={{ background: 'white', padding: '25px', borderRadius: '10px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)', marginBottom: '25px' }}>
                <h3 style={{ marginTop: 0 }}>1. Загрузка датасета (CSV/JSONL)</h3>
                <input type="file" onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)} style={{ marginBottom: '15px', display: 'block' }} />
                <button 
                    onClick={handleUpload} 
                    disabled={!file}
                    style={{ padding: '10px 20px', cursor: file ? 'pointer' : 'not-allowed' }}
                >
                    Загрузить на сервер
                </button>
                {datasetPath && <p style={{ color: 'green', marginTop: '10px' }}>✓ Файл готов к работе: {datasetPath}</p>}
            </section>

            <section style={{ background: 'white', padding: '25px', borderRadius: '10px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)', marginBottom: '25px' }}>
                <h3 style={{ marginTop: 0 }}>2. Настройки обучения</h3>
                
                <div style={{ marginBottom: '15px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}><strong>Базовая модель:</strong></label>
                    <select value={baseModel} onChange={(e) => setBaseModel(e.target.value)} style={{ padding: '8px', width: '100%', maxWidth: '400px' }}>
                        <option value="meta-llama/Llama-2-7b-hf">Llama 2 (7B)</option>
                        <option value="mistralai/Mistral-7B-v0.1">Mistral (7B)</option>
                        <option value="IlyaGusev/