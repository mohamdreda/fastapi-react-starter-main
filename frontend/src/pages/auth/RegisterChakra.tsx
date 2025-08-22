import { useState, type FormEvent, type ChangeEvent } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import {
  Button,
  Input,
  Alert,
  Field,
  VStack,
  HStack,
} from '@chakra-ui/react'
import { useAuth } from '@/context/AuthContext'
import AuthShell from '@/ui/chakra/AuthShell'

export default function RegisterChakra() {
  const { register: registerUser, isLoading, error: globalError } = useAuth()
  const [email, setEmail] = useState('')
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setError(null)
    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }
    const res = await registerUser({
      email,
      first_name: firstName,
      last_name: lastName,
      password,
      confirmPassword,
    })
    if (!res.success) {
      setError(res.error || 'Registration failed')
    }
  }

  return (
    <AuthShell
      title="Create your account"
      subtitle={(
        <>
          Already have an account?{' '}
          <RouterLink to="/login" style={{ color: '#3182CE' }}>
            Sign in
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

          <HStack gap={4} align="stretch" w="full">
            <Field.Root>
              <Field.Label>First name</Field.Label>
              <Input
                type="text"
                value={firstName}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setFirstName(e.target.value)}
                placeholder="First name"
              />
            </Field.Root>
            <Field.Root>
              <Field.Label>Last name</Field.Label>
              <Input
                type="text"
                value={lastName}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setLastName(e.target.value)}
                placeholder="Last name"
              />
            </Field.Root>
          </HStack>

          <Field.Root required>
            <Field.Label>Password</Field.Label>
            <Input
              type="password"
              value={password}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </Field.Root>

          <Field.Root required>
            <Field.Label>Confirm password</Field.Label>
            <Input
              type="password"
              value={confirmPassword}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setConfirmPassword(e.target.value)}
              placeholder="••••••••"
            />
          </Field.Root>

          <Button type="submit" colorScheme="blue" loading={isLoading} w="full">
            Create account
          </Button>
        </VStack>
      </form>
    </AuthShell>
  )
}
