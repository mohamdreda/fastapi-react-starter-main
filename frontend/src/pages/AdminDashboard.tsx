import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/Card'
import { useAuth } from '@/context/AuthContext'
import { useToast } from '@/hooks/use-toast'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { User } from '@/types/user'
import { Flex, Heading, Badge, Table, Button, Box } from '@chakra-ui/react'

type EditUserData = {
  first_name: string
  last_name: string
  email: string
  role: string
}

type NewUserData = {
  first_name: string
  last_name: string
  email: string
  role: string
  password: string
}

export default function AdminDashboard() {
  const [users, setUsers] = useState<User[]>([])
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false)
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)
  const [editData, setEditData] = useState<EditUserData>({ first_name: '', last_name: '', email: '', role: '' })
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
  const [newUser, setNewUser] = useState<NewUserData>({ first_name: '', last_name: '', email: '', role: 'user', password: '' })
  const [page, setPage] = useState(1)
  const pageSize = 10
  const { token, user: authUser } = useAuth()
  const { toast } = useToast()
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

  const fetchUsers = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/users/`, {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        mode: 'cors'
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({} as any))
        throw new Error((errorData as any)?.detail || 'Failed to fetch users')
      }

      const data = await response.json()
      setUsers(data)
    } catch (error) {
      console.error('Failed to fetch users:', error)
    }
  }, [API_URL, token])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  const handleEdit = (user: User) => {
    setSelectedUser(user)
    setEditData({
      first_name: user.first_name,
      last_name: user.last_name,
      email: user.email,
      role: user.role
    })
    setIsEditDialogOpen(true)
  }

  const handleDelete = (user: User) => {
    setSelectedUser(user)
    setIsDeleteDialogOpen(true)
  }

  const saveEdit = async () => {
    if (!selectedUser) return

    try {
      const response = await fetch(`${API_URL}/api/v1/users/${selectedUser.id}`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(editData),
      })

      if (response.ok) {
        toast({
          title: 'Success',
          description: 'User updated successfully',
        })
        await fetchUsers()
        setIsEditDialogOpen(false)
      } else {
        toast({
          variant: 'destructive',
          title: 'Error',
          description: 'Failed to update user',
        })
      }
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Failed to update user',
      })
    }
  }

  const confirmDelete = async () => {
    if (!selectedUser) return

    try {
      const response = await fetch(`${API_URL}/api/v1/users/${selectedUser.id}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })

      if (response.ok) {
        toast({
          title: 'Success',
          description: 'User deleted successfully',
        })
        await fetchUsers()
        setIsDeleteDialogOpen(false)
      } else {
        toast({
          variant: 'destructive',
          title: 'Error',
          description: 'Failed to delete user',
        })
      }
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Failed to delete user',
      })
    }
  }

  const createUser = async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/users/`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newUser),
      })

      if (response.ok) {
        toast({ title: 'Success', description: 'User created successfully' })
        setIsAddDialogOpen(false)
        setNewUser({ first_name: '', last_name: '', email: '', role: 'user', password: '' })
        await fetchUsers()
      } else {
        const err = await response.json().catch(() => ({} as any))
        toast({ variant: 'destructive', title: 'Error', description: (err as any)?.detail || 'Failed to create user' })
      }
    } catch (error) {
      toast({ variant: 'destructive', title: 'Error', description: 'Failed to create user' })
    }
  }

  const totalPages = Math.max(1, Math.ceil(users.length / pageSize))
  const currentPage = Math.min(page, totalPages)
  const startIndex = (currentPage - 1) * pageSize
  const paginatedUsers = users.slice(startIndex, startIndex + pageSize)

  return (
    <>
      <Flex justify="space-between" align="center" mb={6} wrap="wrap" gap={4}>
        <Heading fontSize="2xl" fontWeight="bold">User Management</Heading>
        <Button
                bg="teal.600"
                _hover={{ bg: 'teal.700' }}
                color="white"
                px={6}
                py={2.5}
                borderRadius="md"
                fontWeight="semibold"
                onClick={() => setIsAddDialogOpen(true)}>
          Add User
        </Button>
      </Flex>

      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardDescription>Manage system users</CardDescription>
          </CardHeader>
          <CardContent>
            <Table.Root>
              <Table.Header>
                <Table.Row>
                  <Table.ColumnHeader>Full Name</Table.ColumnHeader>
                  <Table.ColumnHeader>Email</Table.ColumnHeader>
                  <Table.ColumnHeader>Role</Table.ColumnHeader>
                  <Table.ColumnHeader>Status</Table.ColumnHeader>
                  <Table.ColumnHeader>Actions</Table.ColumnHeader>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {paginatedUsers.map((user) => (
                  <Table.Row key={user.id}>
                    <Table.Cell>
                      {`${user.first_name || ''} ${user.last_name || ''}`.trim() || 'N/A'}
                      {authUser?.email === user.email && (
                        <Badge ml={2} colorScheme="teal" variant="subtle">YOU</Badge>
                      )}
                    </Table.Cell>
                    <Table.Cell>{user.email}</Table.Cell>
                    <Table.Cell>
                      <Badge colorScheme="gray" variant="subtle">{user.role === 'admin' ? 'Superuser' : 'User'}</Badge>
                    </Table.Cell>
                    <Table.Cell>
                      <Flex align="center" gap={2}>
                        <Box w={2} h={2} borderRadius="full" bg={(user as any)?.is_active === false ? 'red.500' : 'green.500'} />
                        <span>{(user as any)?.is_active === false ? 'Inactive' : 'Active'}</span>
                      </Flex>
                    </Table.Cell>
                    <Table.Cell>
                      <div className="space-x-2">
                        <Button size="sm" variant="outline" borderColor="blue.500" color="blue.600" _hover={{ bg: 'blue.50' }} onClick={() => handleEdit(user)}>
                          Edit
                        </Button>
                        <Button size="sm" variant="outline" borderColor="red.500" color="red.600" _hover={{ bg: 'red.50' }} onClick={() => handleDelete(user)}>
                          Delete
                        </Button>
                      </div>
                    </Table.Cell>
                  </Table.Row>
                ))}
              </Table.Body>
            </Table.Root>

            <div className="flex items-center justify-between mt-4">
              <div className="text-sm text-gray-500 dark:text-gray-400">Page {currentPage} of {totalPages}</div>
              <div className="space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Add User Dialog */}
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add User</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="new_first_name">First Name</Label>
                <Input
                  id="new_first_name"
                  value={newUser.first_name}
                  onChange={(e) => setNewUser({ ...newUser, first_name: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="new_last_name">Last Name</Label>
                <Input
                  id="new_last_name"
                  value={newUser.last_name}
                  onChange={(e) => setNewUser({ ...newUser, last_name: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="new_email">Email</Label>
                <Input
                  id="new_email"
                  type="email"
                  value={newUser.email}
                  onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="new_role">Role</Label>
                <Select
                  value={newUser.role}
                  onValueChange={(value: string) => setNewUser({ ...newUser, role: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select role" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="user">User</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="new_password">Password</Label>
                <Input
                  id="new_password"
                  type="password"
                  value={newUser.password}
                  onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                />
              </div>
            </div>
            <DialogFooter>
              <Button size="sm" variant="outline" borderColor="gray.300" color="gray.700" bg="gray.50" _hover={{ bg: 'gray.100' }} mr={3} onClick={() => setIsAddDialogOpen(false)}>
                Cancel
              </Button>
              <Button bg="teal.600" _hover={{ bg: 'teal.700' }} color="white" onClick={createUser}>Create</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Edit User Dialog */}
        <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit User</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="first_name">First Name</Label>
                <Input
                  id="first_name"
                  value={editData.first_name}
                  onChange={(e) => setEditData({ ...editData, first_name: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="last_name">Last Name</Label>
                <Input
                  id="last_name"
                  value={editData.last_name}
                  onChange={(e) => setEditData({ ...editData, last_name: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={editData.email}
                  onChange={(e) => setEditData({ ...editData, email: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="role">Role</Label>
                <Select
                  value={editData.role}
                  onValueChange={(value: string) => setEditData({ ...editData, role: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select role" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="user">User</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsEditDialogOpen(false)}>
                Cancel
              </Button>
              <Button bg="teal.600" _hover={{ bg: 'teal.700' }} color="white" onClick={saveEdit}>Save changes</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Confirmation Dialog */}
        <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete User</DialogTitle>
            </DialogHeader>
            <p>Are you sure you want to delete this user? This action cannot be undone.</p>
            <DialogFooter>
              <Button variant="outline" borderColor="gray.400" _hover={{ bg: 'gray.50' }} onClick={() => setIsDeleteDialogOpen(false)}>
                Cancel
              </Button>
              <Button bg="red.600" _hover={{ bg: 'red.700' }} color="white" onClick={confirmDelete}>
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
    </>
  )
}
