import { ReactNode } from 'react'
import { ChakraProvider, defaultSystem } from '@chakra-ui/react'

interface Props {
  children: ReactNode
}

export default function ChakraScoped({ children }: Props) {
  return <ChakraProvider value={defaultSystem}>{children}</ChakraProvider>
}
