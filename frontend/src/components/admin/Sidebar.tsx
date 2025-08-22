import { Box, Flex, Icon, Text } from '@chakra-ui/react'
import { NavLink } from 'react-router-dom'
import { FiHome, FiBriefcase, FiSettings, FiUsers } from 'react-icons/fi'
import Logo from '@/assets/data-cleaning-logo.svg'

const items = [
  { icon: FiHome, title: 'Dashboard', path: '/admin/dashboard' },
  { icon: FiBriefcase, title: 'Items', path: '/admin/items' },
  { icon: FiSettings, title: 'User Settings', path: '/admin/settings' },
  { icon: FiUsers, title: 'Admin', path: '/admin/dashboard' },
]

export default function Sidebar() {
  return (
    <Box display={{ base: 'none', md: 'flex' }} position="sticky" bg="bg.subtle" top={0} minW="xs" h="100vh" p={4}>
      <Box w="100%">
        <Flex align="center" px={4} py={2} gap={2}>
          <img src={Logo} alt="Data Cleaning" style={{ width: 24, height: 24 }} />
          <Text fontWeight="bold" fontSize="lg">Data Cleaning</Text>
        </Flex>
        <Text fontSize="xs" px={4} py={2} fontWeight="bold" color="fg.muted">
          Menu
        </Text>
        <Box>
          {items.map(({ icon, title, path }) => (
            <NavLink key={title} to={path}>
              {({ isActive }) => (
                <Flex
                  gap={4}
                  px={4}
                  py={2}
                  _hover={{ bg: 'bg.subtle' }}
                  bg={isActive ? 'bg.subtle' : 'transparent'}
                  alignItems="center"
                  fontSize="sm"
                  borderRadius="md"
                >
                  <Icon as={icon} alignSelf="center" />
                  <Text ml={2}>{title}</Text>
                </Flex>
              )}
            </NavLink>
          ))}
        </Box>
      </Box>
    </Box>
  )
}
