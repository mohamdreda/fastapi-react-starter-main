// src/components/ui/PageLoader.tsx
import { Spinner } from './Spinner';

export function PageLoader() {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-background">
      <Spinner size="lg" className="text-primary" />
      <span className="sr-only">Loading...</span>
    </div>
  );
}