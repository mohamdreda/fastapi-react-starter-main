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

interface Props {
  missingBefore: Record<string, number>;
  missingAfter: Record<string, number>;
}

const MissingValueChart: React.FC<Props> = ({ missingBefore, missingAfter }) => {
  const columns = Object.keys(missingBefore);
  if (columns.length === 0) return null;

  const data = {
    labels: columns,
    datasets: [
      {
        label: 'Missing Before',
        data: columns.map((c) => missingBefore[c]),
        backgroundColor: 'rgba(239,68,68,0.6)',
      },
      {
        label: 'Missing After',
        data: columns.map((c) => missingAfter[c]),
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
      x: { ticks: { maxRotation: 90, minRotation: 45 } },
    },
  } as const;

  const totalBefore = columns.reduce((sum, c) => sum + missingBefore[c], 0);
  const totalAfter = columns.reduce((sum, c) => sum + missingAfter[c], 0);

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-2">Missing Values Before vs After</h3>
      <div className="text-sm mb-2 text-gray-600">Total before: {totalBefore.toLocaleString()} • Total after: {totalAfter.toLocaleString()} • Filled: {(totalBefore-totalAfter).toLocaleString()}</div>
      <Bar data={data} options={options} />
    </div>
  );
};

export default MissingValueChart;
