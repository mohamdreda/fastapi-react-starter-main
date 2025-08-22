import { Outlet, useNavigate } from 'react-router-dom'
import { Box, Flex, Button, Container } from '@chakra-ui/react'
import { useAuth } from '@/context/AuthContext'
import AdminSidebar from '@/components/admin/Sidebar'

export default function AdminLayout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  

  return (

      <Flex minH="100vh" bg="bg.canvas">
        {/* Sidebar */}
        <AdminSidebar />

        {/* Content */}
        <Flex direction="column" flex={1} minW={0}>
          <Flex as="header" align="center" justify="flex-end" px={6} py={3} borderBottomWidth="1px" bg="bg.panel" gap={4}>
            <Box w={8} h={8} borderRadius="full" bg="teal.500" color="white" display="flex" alignItems="center" justifyContent="center" fontSize="xs" fontWeight="bold">{(user?.email || 'U').slice(0,1).toUpperCase()}</Box>
            <Button size="sm" variant="outline" borderColor="gray.300" _hover={{ bg: 'gray.50' }} onClick={() => {
              logout();
              navigate('/login');
            }}>
              Logout
            </Button>
          </Flex>
          <Box as="main" flex={1} py={6}>
            <Container maxW="7xl">
              <Outlet />
            </Container>
          </Box>
        </Flex>
      </Flex>

  )
}
