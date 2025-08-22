import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface DimensionalityReductionPlotProps {
  data: Array<{
    [key: string]: number;
  }>;
  xKey?: string;
  yKey?: string;
}

const DimensionalityReductionPlot: React.FC<DimensionalityReductionPlotProps> = ({ 
  data, 
  xKey = 'latent_0', 
  yKey = 'latent_1' 
}) => {
  if (!data || data.length === 0) return null;
  
  // Ensure the data has the required keys
  if (!data[0].hasOwnProperty(xKey) || !data[0].hasOwnProperty(yKey)) {
    return (
      <div className="text-red-500 p-4 bg-red-50 rounded">
        Cannot render plot: Data does not contain required dimensions ({xKey}, {yKey})
      </div>
    );
  }

  // Format data for the scatter plot
  const formattedData = data.map(item => ({
    x: item[xKey],
    y: item[yKey]
  }));

  return (
    <div className="w-full h-80 bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-medium mb-4">2D Visualization of Extracted Features</h3>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart
          margin={{
            top: 20,
            right: 20,
            bottom: 20,
            left: 20,
          }}
        >
          <CartesianGrid />
          <XAxis 
            type="number" 
            dataKey="x" 
            name={xKey} 
            label={{ value: xKey, position: 'insideBottom', offset: -10 }} 
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name={yKey} 
            label={{ value: yKey, angle: -90, position: 'insideLeft' }} 
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }} 
            formatter={(value: any) => [Number(value).toFixed(4)]} 
            labelFormatter={() => ''} 
          />
          <Scatter name="Features" data={formattedData} fill="#8884d8" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DimensionalityReductionPlot;
