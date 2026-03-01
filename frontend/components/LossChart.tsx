import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ChartProps {
    data: Array<{ step: number; loss: number }>;
}

export default function LossChart({ data }: ChartProps) {
    return (
        <div style={{ width: '100%', height: 400 }}>
            <ResponsiveContainer>
                <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" name="Шаг" />
                    <YAxis name="Функция потерь" />
                    <Tooltip />
                    <Legend payload={[{ value: 'Потери при обучении', type: 'line', color: '#8884d8' }]} />
                    <Line type="monotone" dataKey="loss" stroke="#8884d8" activeDot={{ r: 8 }} isAnimationActive={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}