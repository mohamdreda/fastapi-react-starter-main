import { useState, type FormEvent, type ChangeEvent } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import {
  Button,
  Input,
  Alert,
  Field,
  VStack,
} from '@chakra-ui/react'
import { useAuth } from '@/context/AuthContext'
import AuthShell from '@/ui/chakra/AuthShell'

export default function LoginChakra() {
  const { login, isLoading, error: globalError } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setError(null)
    const res = await login({ email, password })
    if (!res.success) {
      setError(res.error || 'Login failed')
    }
  }

  return (
    <AuthShell
      title="Sign in to your account"
      subtitle={(
        <>
          Or{' '}
          <RouterLink to="/register" style={{ color: '#3182CE' }}>
            create a new account
          </RouterLink>
        </>
      )}
    >
      {(error || globalError) && (
        <Alert.Root status="error">
          <Alert.Indicator />
          <Alert.Description>{error || globalError}</Alert.Description>
        </Alert.Root>
      )}

      <form onSubmit={onSubmit}>
        <VStack gap={4}>
          <Field.Root required>
            <Field.Label>Email</Field.Label>
            <Input
              type="email"
              value={email}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </Field.Root>

          <Field.Root required>
            <Field.Label>Password</Field.Label>
            <Input
              type="password"
              value={password}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </Field.Root>

          <Button type="submit" colorScheme="blue" loading={isLoading} w="full">
            Sign In
          </Button>
        </VStack>
      </form>
    </AuthShell>
  )
}
