import { useEffect, useMemo, useState } from 'react'
import { useColorMode } from '@chakra-ui/color-mode'
import { useToast } from '@/hooks/use-toast'
import { useTheme } from 'next-themes'
import { useAuth } from '@/context/AuthContext'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function AdminSettings() {
  const { token, user: authUser } = useAuth()
  const { colorMode, setColorMode } = useColorMode()
  const { theme, setTheme } = useTheme()
  const { toast } = useToast()

  const storedTheme = localStorage.getItem('theme')
  const prefersDark = storedTheme ? storedTheme === 'dark' : window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  const [isDark, setIsDark] = useState(prefersDark)

  useEffect(() => {
    if (isDark) {
      setTheme('dark')
      if (colorMode !== 'dark' && typeof setColorMode === 'function') setColorMode('dark')
    } else {
      setTheme('light')
      if (colorMode !== 'light' && typeof setColorMode === 'function') setColorMode('light')
    }
  }, [isDark])

  const initial = useMemo(() => {
    const fullName = `${authUser?.first_name || ''} ${authUser?.last_name || ''}`.trim()
    return {
      fullName: fullName || (authUser?.email?.split('@')[0] ?? ''),
      email: authUser?.email || '',
    }
  }, [authUser])

  const [fullName, setFullName] = useState<string>(initial.fullName)
  const [email, setEmail] = useState<string>(initial.email)

  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [activeTab, setActiveTab] = useState<'profile' | 'password' | 'appearance'>('profile')
  const [confirmPassword, setConfirmPassword] = useState('')

  const onResetProfile = () => {
    setFullName(initial.fullName)
    setEmail(initial.email)
  }

  const onSaveProfile = async () => {
    if (!authUser) return
    try {
      const [first_name, ...rest] = fullName.trim().split(/\s+/)
      const last_name = rest.join(' ')
      const resp = await fetch(`${API_URL}/api/v1/users/${authUser.id}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ first_name, last_name, email }),
        mode: 'cors',
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({} as any))
        throw new Error((err as any)?.detail || 'Failed to save profile')
      }
      // Update cached user in localStorage so the change persists across reloads
      const updated = { ...authUser, first_name, last_name, email }
      localStorage.setItem('user', JSON.stringify(updated))
      toast({ title: 'Saved', description: 'Profile updated successfully' })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to save profile'
      toast({ variant: 'destructive', title: 'Error', description: msg })
    }
  }

  const onRequestPasswordReset = async () => {
    try {
      const emailParam = encodeURIComponent(authUser?.email || '')
      const resp = await fetch(`${API_URL}/api/v1/auth/request-password-reset?email=${emailParam}`, {
        method: 'POST',
      })
      if (!resp.ok) {
        throw new Error('Failed to request password reset')
      }
      toast({ title: 'Password reset', description: 'If your account exists, a reset link will be sent.' })
    } catch (e) {
      toast({ variant: 'destructive', title: 'Error', description: 'Could not request password reset' })
    }
  }

  const onChangePassword = async () => {
    if (currentPassword.length === 0) {
    toast({ variant: 'destructive', title: 'Error', description: 'Current password is required' })
    return
  }
  if (newPassword.length < 6) {
      toast({ variant: 'destructive', title: 'Error', description: 'Password must be at least 6 characters' })
      return
    }
    if (newPassword !== confirmPassword) {
      toast({ variant: 'destructive', title: 'Error', description: 'Passwords do not match' })
      return
    }
    try {
      const body = new URLSearchParams({ old_password: currentPassword, new_password: newPassword })
      const resp = await fetch(`${API_URL}/api/v1/auth/change-password`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: body.toString(),
        mode: 'cors',
      })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({} as any))
        throw new Error((err as any)?.detail || 'Failed to change password')
      }
      toast({ title: 'Password updated', description: 'You can now use your new password.' })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to change password'
      toast({ variant: 'destructive', title: 'Error', description: msg })
    }
  }

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-semibold mb-6">User Settings</h2>

      {/* Tab buttons */}
      <div className="flex gap-3 mb-6">
        {(['profile', 'password', 'appearance'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={
              `px-4 py-2 rounded-md border ${activeTab === tab ? 'bg-teal-600 text-white' : 'bg-transparent text-teal-600'}`
            }
          >
            {tab === 'profile' ? 'My profile' : tab === 'password' ? 'Password' : 'Appearance'}
          </button>
        ))}
      </div>

      {/* Profile Tab */}
      {activeTab === 'profile' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium mb-3">User Information</h3>
            <div className="space-y-4">
              <div className="space-y-1">
                <label className="text-sm font-medium block">Full name</label>
                <input
                  className="w-full border rounded p-2"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium block">Email</label>
                <input
                  className="w-full border rounded p-2 bg-gray-100 dark:bg-gray-800"
                  type="email"
                  value={email}
                  disabled
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>
          </div>
          <div className="flex gap-3">
            <button className="px-4 py-2 bg-teal-600 text-white rounded" onClick={onSaveProfile}>Save</button>
            <button className="px-4 py-2 border rounded" onClick={onResetProfile}>Cancel</button>
          </div>
        </div>
      )}

      {/* Password Tab */}
      {activeTab === 'password' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium mb-3">Change Password</h3>
            <div className="space-y-4">
              <div className="space-y-1">
                <label className="text-sm font-medium block">Current password</label>
                <input
                  type="password"
                  className="w-full border rounded p-2"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium block">New password</label>
                <input
                  type="password"
                  className="w-full border rounded p-2"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium block">Confirm password</label>
                <input
                  type="password"
                  className="w-full border rounded p-2"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>
            </div>
          </div>
          <div className="flex gap-3">
            <button className="px-4 py-2 bg-teal-600 text-white rounded" onClick={onChangePassword}>Save</button>
            <button
              className="px-4 py-2 border rounded"
              onClick={() => { setNewPassword(''); setConfirmPassword('') }}
            >
              Cancel
            </button>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Tip: We send a reset link using your email to change the password securely.</p>
        </div>
      )}

      {/* Appearance Tab */}
      {activeTab === 'appearance' && (
        <div className="space-y-6">
          <h3 className="text-lg font-medium mb-3">Theme</h3>
          <div className="flex items-center gap-3">
            <input
              id="dark-toggle"
              type="checkbox"
              checked={isDark}
              onChange={() => setIsDark(!isDark)}
              className="h-5 w-5"
            />
            <label htmlFor="dark-toggle">Dark mode</label>
          </div>
        </div>
      )}
    </div>
  )
}
