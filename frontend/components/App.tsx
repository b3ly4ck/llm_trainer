import React, { useState, useEffect, useRef } from 'react';
import LossChart from './components/LossChart';

export default function App() {
    const [taskId, setTaskId] = useState<number | null>(null);
    const [status, setStatus] = useState<string>('Ожидание');
    const [metricsData, setMetricsData] = useState<Array<{ step: number; loss: number }>>([]);
    const [error, setError] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const startTraining = async () => {
        try {
            // Хардкод пути для примера. В реальности путь берется из ответа эндпоинта /api/dataset/upload
            const response = await fetch('http://localhost:8000/api/train?dataset_path=processed_data/dataset.jsonl', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ epochs: 3, batch_size: 4, lr: 0.0002 })
            });
            const data = await response.json();
            
            setTaskId(data.task_id);
            setStatus(data.status);
            setMetricsData([]);
            setError(null);
        } catch (err: any) {
            setError('Ошибка запуска: ' + err.message);
        }
    };

    useEffect(() => {
        if (!taskId) return;

        // Подключение к WebSocket при получении ID задачи
        const ws = new WebSocket(`ws://localhost:8000/ws/metrics/${taskId}`);
        wsRef.current = ws;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.status) setStatus(data.status);
            if (data.error) setError(data.error);
            
            // Если пришли метрики, добавляем их в массив для графика
            if (data.loss !== undefined && data.step !== undefined) {
                setMetricsData(prev => [...prev, { step: data.step, loss: data.loss }]);
            }
        };

        ws.onclose = () => console.log('Соединение WebSocket закрыто');

        return () => {
            if (wsRef.current) wsRef.current.close();
        };
    }, [taskId]);

    return (
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto', fontFamily: 'sans-serif' }}>
            <h1>Платформа Fine-Tuning LLM</h1>
            
            <div style={{ marginBottom: '20px' }}>
                <button 
                    onClick={startTraining} 
                    disabled={status === 'Running'}
                    style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}
                >
                    Запустить обучение
                </button>
            </div>

            <div style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
                <p><strong>ID задачи:</strong> {taskId || '—'}</p>
                <p><strong>Статус:</strong> {status}</p>
                {error && <p style={{ color: 'red' }}><strong>Ошибка:</strong> {error}</p>}
            </div>

            <h2>График обучения</h2>
            {metricsData.length > 0 ? (
                <LossChart data={metricsData} />
            ) : (
                <p style={{ color: '#666' }}>Нет данных для отображения графика. Запустите задачу.</p>
            )}
        </div>
    );
}