import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface LossCurveProps {
  metrics: {
    epochs: number[];
    loss: number[];
    val_loss: number[];
  };
}

const LossCurve: React.FC<LossCurveProps> = ({ metrics }) => {
  if (!metrics || !metrics.loss) return null;
  const data = metrics.epochs.map((epoch, i) => ({
    epoch,
    loss: metrics.loss[i],
    val_loss: metrics.val_loss[i],
  }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <XAxis dataKey="epoch" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Train Loss" />
        <Line type="monotone" dataKey="val_loss" stroke="#82ca9d" name="Val Loss" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default LossCurve;
