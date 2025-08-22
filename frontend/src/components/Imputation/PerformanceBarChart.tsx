import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Legend,
  Tooltip,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Legend, Tooltip);

interface PerfMetrics {
  runtime_seconds: number;
  rmse?: number;
  mae?: number;
}
interface HistoryItem {
  strategy: string;
  performance: PerfMetrics;
}
interface Props {
  history: HistoryItem[];
}

const PerformanceBarChart: React.FC<Props> = ({ history }) => {
  if (history.length === 0) return null;

  // Aggregate latest entry per strategy
  const latestByAlgo: Record<string, PerfMetrics> = {};
  for (const h of history) {
    latestByAlgo[h.strategy] = h.performance;
  }

  const labels = Object.keys(latestByAlgo);
  const runtimes = labels.map((k) => latestByAlgo[k].runtime_seconds);
  const rmses = labels.map((k) => latestByAlgo[k].rmse ?? 0);
  const maes = labels.map((k) => latestByAlgo[k].mae ?? 0);

  const data = {
    labels,
    datasets: [
      {
        label: 'Runtime (s)',
        data: runtimes,
        backgroundColor: 'rgba(59,130,246,0.6)',
      },
      {
        label: 'RMSE',
        data: rmses,
        backgroundColor: 'rgba(234,88,12,0.6)',
      },
      {
        label: 'MAE',
        data: maes,
        backgroundColor: 'rgba(16,185,129,0.6)',
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' as const },
    },
    scales: {
      y: { beginAtZero: true },
    },
  } as const;

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-2">Algorithm Performance Comparison</h3>
      <Bar data={data} options={options} />
    </div>
  );
};

export default PerformanceBarChart;
