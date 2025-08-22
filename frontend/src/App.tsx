import { RouterProvider } from "react-router-dom";
import { router } from "./routes/router";
import { ThemeProvider } from "./components/ui/ThemeProvider";
import ChakraScoped from '@/ui/chakra/ChakraScoped';
import ErrorBoundary from "./components/ErrorBoundary";

export default function App() {
  return (
    <ErrorBoundary>
      <ChakraScoped>
        <ThemeProvider defaultTheme="light">
          <RouterProvider router={router} />
      </ThemeProvider>
      </ChakraScoped>
    </ErrorBoundary>
  );
}