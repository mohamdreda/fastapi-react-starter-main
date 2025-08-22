import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { useAuth } from '../context/AuthContext';
import HeroIllustration from '@/assets/hero-illustration.svg';

// Features grid removed per request

export default function Home() {
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuth();

  return (
    <div className="space-y-12 py-8">
      <header className="text-center space-y-4">
        <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-gray-900 dark:text-white">
          Data Cleaning <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-cyan-400 to-indigo-400">Platform</span>
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Upload, clean, and prepare your data with modern tools.
        </p>
        <div className="flex justify-center gap-3 pt-2">
          <Button onClick={() => navigate('/register')} variant="brand">
            Get Started Free
          </Button>
          <Button
            onClick={() => document.getElementById('tools')?.scrollIntoView({ behavior: 'smooth' })}
            variant="secondary"
          >
            Explore Tools
          </Button>
        </div>
        <div className="mt-8 flex justify-center">
          <img
            src={HeroIllustration}
            alt="Data cleaning platform illustration"
            className="w-full max-w-5xl rounded-xl border border-gray-200 dark:border-gray-700 shadow-xl"
            loading="lazy"
          />
        </div>
      </header>

      

      {/* What is Data Cleaning? */}
      <section id="what-is-data-cleaning" className="grid gap-4">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">What is Data Cleaning?</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300">
          Data cleaning improves data quality by fixing missing values, removing duplicates, correcting
          inconsistent formats, and handling outliers. Clean data leads to clearer analysis, stronger
          decisions, and better model performance.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            { title: 'Quality', text: 'Correct errors and standardize formats.' },
            { title: 'Trust', text: 'Remove duplicates and inconsistencies.' },
            { title: 'Performance', text: 'Prepare data for analysis and ML.' },
          ].map((i) => (
            <div key={i.title} className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
              <h4 className="font-medium text-gray-900 dark:text-white">{i.title}</h4>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">{i.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Tools included */}
      <section id="tools" className="grid gap-6">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">Tools included</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { title: 'Outlier Detection', desc: 'Detect anomalies and extreme values in your datasets.', tag: 'User' },
            { title: 'Deduplication', desc: 'Find and merge duplicate records with flexible blocking.', tag: 'User' },
            { title: 'Imputation', desc: 'Fill missing values using multiple strategies and previews.', tag: 'User' },
            { title: 'Preprocessing', desc: 'Standardize, normalize, and encode categorical variables.', tag: 'User' },
            { title: 'Clustering & Similarity', desc: 'Group similar records and compute similarity scores.', tag: 'User' },
          ].map((item) => (
            <div
              key={item.title}
              className="p-5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:shadow-md transition-all"
            >
              <div className="flex items-start justify-between">
                <h4 className="font-medium text-gray-900 dark:text-white">{item.title}</h4>
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">
                  {item.tag}
                </span>
              </div>
              <p className="text-sm mt-2 text-gray-600 dark:text-gray-300">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* About Us */}
      <section id="about" className="grid gap-6">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">About Us</h3>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white">Our Mission</h4>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
              We help teams clean and prepare data faster with an intuitive, secure platform. From detecting
              outliers to resolving duplicates and exporting golden records, our tools are built for speed,
              accuracy, and collaboration.
            </p>
            <ul className="mt-4 space-y-2 text-sm text-gray-600 dark:text-gray-300 list-disc list-inside">
              <li>Modern UX with light and dark themes</li>
              <li>FastAPI backend with robust APIs</li>
              <li>Role-based dashboards for Admins and Users</li>
            </ul>
          </div>
          <div className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-gradient-to-br from-emerald-500/10 via-cyan-500/10 to-indigo-600/10">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white">Our Values</h4>
            <ul className="mt-3 space-y-2 text-sm text-gray-700 dark:text-gray-200">
              <li><span className="font-medium">Clarity</span> — clean UI and results you can trust</li>
              <li><span className="font-medium">Speed</span> — fast workflows and responsive UI</li>
              <li><span className="font-medium">Security</span> — JWT auth and scoped access</li>
            </ul>
            <div className="mt-4">
              <Button variant="brand" className="w-full" onClick={() => navigate('/register')}>
                Create your free account
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="grid gap-6">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">How it works</h3>
        <ol className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { step: '1', title: 'Upload', desc: 'Import CSV files and preview your data securely.' },
            { step: '2', title: 'Clean & Transform', desc: 'Run outlier detection, deduplication, imputation, and transformations.' },
            { step: '3', title: 'Review & Export', desc: 'Validate results and download clean data and reports.' },
          ].map((s) => (
            <li key={s.step} className="p-5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
              <div className="flex items-center gap-3">
                <span className="h-7 w-7 rounded-full bg-indigo-600 text-white grid place-items-center text-sm font-semibold">
                  {s.step}
                </span>
                <h4 className="font-medium text-gray-900 dark:text-white">{s.title}</h4>
              </div>
              <p className="text-sm mt-2 text-gray-600 dark:text-gray-300">{s.desc}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* Built with */}
      <section id="tech" className="grid gap-6">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">Built with</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300">
          This app uses a modern stack designed for speed, reliability, and a great developer experience.
        </p>
        <div className="flex flex-wrap gap-2">
          {[ 'FastAPI', 'React 19', 'TypeScript', 'Tailwind CSS', 'Vite', 'React Router', 'JWT Auth' ].map((t) => (
            <span
              key={t}
              className="px-3 py-1 rounded-full text-xs border bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-800 dark:text-gray-200 dark:border-gray-700"
            >
              {t}
            </span>
          ))}
        </div>
      </section>

      {/* Dashboards */}
      <section className="grid gap-6">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">Dashboards</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <div className="flex items-center justify-between">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white">User Dashboard</h4>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">Horizon UI</span>
            </div>
            <ul className="mt-3 space-y-2 text-sm text-gray-600 dark:text-gray-300 list-disc list-inside">
              <li>Upload files and explore datasets</li>
              <li>Run outlier detection, deduplication, imputation</li>
              <li>Clustering and similarity scoring</li>
              <li>Data transformation: standardization and categorical encoding</li>
            </ul>
          </div>
          <div className="p-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
            <div className="flex items-center justify-between">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white">Admin Dashboard</h4>
              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300">Notus React</span>
            </div>
            <ul className="mt-3 space-y-2 text-sm text-gray-600 dark:text-gray-300 list-disc list-inside">
              <li>User and role management</li>
              <li>System settings and access control</li>
              <li>Activity insights and status monitoring</li>
              <li>Secure JWT-based authentication</li>
            </ul>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="grid gap-4">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">FAQ</h3>
        <details className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <summary className="cursor-pointer font-medium text-gray-900 dark:text-white">Do I need an account?</summary>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">Yes. Create a free account to access all tools and save your work.</p>
        </details>
        <details className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <summary className="cursor-pointer font-medium text-gray-900 dark:text-white">Is dark mode supported?</summary>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">Yes. The UI supports light and dark themes with automatic persistence.</p>
        </details>
        <details className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <summary className="cursor-pointer font-medium text-gray-900 dark:text-white">Which stack is used?</summary>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">FastAPI backend with React 19, Tailwind CSS, and Chakra UI components.</p>
        </details>
      </section>

      <div className="flex justify-center flex-wrap gap-3">
        <Button onClick={() => navigate('/about')} variant="default">
          Learn More
        </Button>
        {isAuthenticated ? (
          <Button
            onClick={() =>
              navigate(user?.role === 'admin' ? '/admin/dashboard' : `/user/dashboard/${user?.id}`)
            }
            variant="secondary"
          >
            Go to Dashboard
          </Button>
        ) : (
          <Button onClick={() => navigate('/register')} variant="secondary">
            Get Started
          </Button>
        )}
      </div>
    </div>
  );
}