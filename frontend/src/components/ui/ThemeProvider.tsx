import { ReactNode } from 'react';
import { ThemeProvider as NextThemeProvider } from 'next-themes';

interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: 'light' | 'dark';
}

export function ThemeProvider({ children, defaultTheme = 'light' }: ThemeProviderProps) {
  return (
    <NextThemeProvider attribute="class" defaultTheme={defaultTheme}>
      {children}
    </NextThemeProvider>
  );
}
