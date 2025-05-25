import { useNavigate } from 'react-router-dom';
import { CircleCheck, Sparkles } from 'lucide-react';
import { Button } from '../components/ui/Button';

const features = [
  { id: '1', text: 'FastAPI Backend with Health Check' },
  { id: '2', text: 'React 19 with Modern Patterns' },
  { id: '3', text: 'Native Fetch API Integration' },
  { id: '4', text: 'Modern Data Fetching' },
  { id: '5', text: 'Tailwind CSS with Dark Mode' },
  { id: '6', text: 'Responsive Design' },
  { id: '7', text: 'Error Boundaries' },
  { id: '8', text: 'Docker Support' },
];

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="space-y-12 py-8">
      {/* En-tête inchangé */}
      
      <div className="grid gap-6">
        <div className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm hover:shadow-md transition-all">
          <div className="flex items-start space-x-4">
            <div className="flex-shrink-0 p-2 rounded-lg bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
              <CircleCheck />
            </div>
            <div className="flex-1">
              <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                Backend Status
              </h2>
              <div className="text-green-500">Operational</div>
            </div>
          </div>
        </div>

        {/* Section Features inchangée */}
      </div>

      <div className="flex justify-center space-x-4">
        <Button onClick={() => navigate('/about')} variant="default">
          Learn More
        </Button>
        <Button onClick={() => navigate('/dashboard')} variant="secondary">
          View Dashboard
        </Button>
      </div>
    </div>
  );
}