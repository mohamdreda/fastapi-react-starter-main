import React, { useEffect, useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/Button';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface MeResponse {
  id: number | string;
  email: string;
  first_name: string;
  last_name: string;
  role: 'user' | 'admin' | 'moderator';
  company?: string | null;
  phone_number?: string | null;
  created_at?: string;
}

export default function UserProfilePage() {
  const { token, user } = useAuth();
  const [data, setData] = useState<MeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;
    async function load() {
      if (!token) return;
      setLoading(true);
      setError(null);
      try {
        const resp = await fetch(`${API_URL}/api/v1/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const json = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(json.detail || 'Failed to load profile');
        if (!ignore) setData(json);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load profile');
      } finally {
        if (!ignore) setLoading(false);
      }
    }
    load();
    return () => { ignore = true; };
  }, [token]);

  return (
    <div className="container mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Your Profile</h1>

      {error && <div className="text-red-600 dark:text-red-400 mb-4 text-sm">{error}</div>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Account</h2>
          <div className="space-y-2 text-gray-700 dark:text-gray-200">
            <div><span className="font-medium">Name:</span> {data?.first_name} {data?.last_name}</div>
            <div><span className="font-medium">Email:</span> {data?.email}</div>
            <div><span className="font-medium">Role:</span> {data?.role}</div>
            <div><span className="font-medium">Member since:</span> {data?.created_at ? new Date(data.created_at).toLocaleDateString() : '—'}</div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Profile</h2>
          <div className="space-y-2 text-gray-700 dark:text-gray-200">
            <div><span className="font-medium">Company:</span> {data?.company || '—'}</div>
            <div><span className="font-medium">Phone:</span> {data?.phone_number || '—'}</div>
          </div>
          {user && (
            <div className="mt-4">
              <Link to={`/user/dashboard/${user.id}/settings`}>
                <Button variant="brand">Edit profile</Button>
              </Link>
            </div>
          )}
        </div>
      </div>

      <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Usage statistics</h2>
        <p className="text-gray-600 dark:text-gray-300">Coming soon…</p>
      </div>

      {loading && <div className="text-xs mt-4 text-gray-500">Loading…</div>}
    </div>
  );
}
