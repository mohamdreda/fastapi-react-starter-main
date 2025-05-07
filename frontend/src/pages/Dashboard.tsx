import { Suspense } from 'react'
import { ChartColumnIncreasing, Clock1, Zap } from 'lucide-react'
import { useAppDispatch } from '../context/AppContext'
import { showNotification } from '../context/AppContext'

export default function Dashboard() {
  const dispatch = useAppDispatch()

  const handleCardClick = (title: string) => {
    showNotification(dispatch, `Clicked ${title} card`, 'info')
  }

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4 text-gray-900 dark:text-white">Dashboard</h1>
        <p className="text-lg text-gray-600 dark:text-gray-300">
          Monitor your application metrics and performance
        </p>
      </div>

      <Suspense
        fallback={
          <div className="text-center py-8 text-gray-600 dark:text-gray-400">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4" />
            Loading dashboard data...
          </div>
        }
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <DashboardCard
            title="Statistics"
            description="View key metrics and analytics"
            icon={<ChartColumnIncreasing />}
            onClick={handleCardClick}
          />
          <DashboardCard
            title="Recent Activity"
            description="Track the latest updates and changes"
            icon={<Clock1 />}
            onClick={handleCardClick}
          />
          <DashboardCard
            title="Performance"
            description="Monitor system performance metrics"
            icon={<Zap />}
            onClick={handleCardClick}
          />
        </div>
      </Suspense>
    </div>
  )
}

function DashboardCard({
  title,
  description,
  icon,
  onClick,
}: {
  title: string
  description: string
  icon: React.ReactNode
  onClick: (title: string) => void
}) {
  return (
    <div
      className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 
        bg-white dark:bg-gray-800 shadow-sm hover:shadow-md transition-all 
        cursor-pointer group"
      onClick={() => onClick(title)}
    >
      <div className="flex items-start space-x-4">
        <div
          className="flex-shrink-0 p-2 rounded-lg bg-blue-50 dark:bg-blue-900/30 
          text-blue-600 dark:text-blue-400 group-hover:bg-blue-100 
          dark:group-hover:bg-blue-900/50 transition-colors"
        >
          {icon}
        </div>
        <div>
          <h2
            className="text-xl font-semibold mb-2 text-gray-900 dark:text-white group-hover:text-blue-600 
            dark:group-hover:text-blue-400 transition-colors"
          >
            {title}
          </h2>
          <p className="text-gray-600 dark:text-gray-300">{description}</p>
        </div>
      </div>
    </div>
  )
}
