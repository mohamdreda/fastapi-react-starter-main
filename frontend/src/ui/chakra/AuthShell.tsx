import { ReactNode } from 'react'
import { Box, Container, Heading, Text, VStack } from '@chakra-ui/react'

interface AuthShellProps {
  title: string
  subtitle?: ReactNode
  children: ReactNode
}

export default function AuthShell({ title, subtitle, children }: AuthShellProps) {
  return (
    <Box minH="100svh" bgGradient="linear(to-b, gray.50, white)" _dark={{ bg: 'gray.900' }} py={16}>
      <Container maxW="md">
        <VStack gap={6} align="stretch">
          <VStack gap={1}>
            <Heading size="lg" textAlign="center">{title}</Heading>
            {subtitle ? (
              <Text textAlign="center" color="gray.500">{subtitle}</Text>
            ) : null}
          </VStack>

          <Box borderWidth="1px" borderColor="gray.200" _dark={{ borderColor: 'gray.700', bg: 'gray.800' }} bg="white" p={6} rounded="lg" shadow="md">
            {children}
          </Box>
        </VStack>
      </Container>
    </Box>
  )
}
