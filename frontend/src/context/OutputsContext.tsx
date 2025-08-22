import React, { createContext, useContext, useState } from 'react';
// OutputSidebar UI removed; define a minimal local type to keep context decoupled
type Output = unknown;

interface OutputsContextType {
  outputs: Output[];
  addOutput: (output: Output) => void;
  clearOutputs: () => void;
}

const OutputsContext = createContext<OutputsContextType | undefined>(undefined);

export const useOutputs = () => {
  const context = useContext(OutputsContext);
  if (context === undefined) {
    throw new Error('useOutputs must be used within an OutputsProvider');
  }
  return context;
};

interface OutputsProviderProps {
  children: React.ReactNode;
}

export const OutputsProvider: React.FC<OutputsProviderProps> = ({ children }) => {
  const [outputs, setOutputs] = useState<Output[]>([]);

  const addOutput = (output: Output) => {
    setOutputs(prev => [...prev, output]);
  };

  const clearOutputs = () => {
    setOutputs([]);
  };

  return (
    <OutputsContext.Provider value={{ outputs, addOutput, clearOutputs }}>
      {children}
    </OutputsContext.Provider>
  );
};
