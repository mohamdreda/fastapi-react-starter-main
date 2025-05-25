import React from 'react';
import { FaArrowLeft } from 'react-icons/fa';

export const WelcomeDashboard: React.FC = () => {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center space-y-6 max-w-2xl mx-auto p-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Welcome to Your Data Cleaning Workspace
        </h1>
        
        <p className="text-lg text-gray-600">
          Select a tool from the left sidebar to get started
        </p>

        <div className="flex items-center justify-center text-gray-500 mt-8">
          <FaArrowLeft className="w-6 h-6 mr-2 animate-pulse" />
          <span>Choose a tool</span>
        </div>
      </div>
    </div>
  );
};
