import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
} from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';

interface ConfirmationDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title?: string;
  message?: string | React.ReactNode;
  confirmButtonText?: string;
  cancelButtonText?: string;
  confirmButtonDisabled?: boolean;
  cancelButtonDisabled?: boolean;
  confirmButtonColor?: 'primary' | 'secondary' | 'error' | 'success' | 'info' | 'warning';
}

const ConfirmationDialog: React.FC<ConfirmationDialogProps> = ({
  open,
  onClose,
  onConfirm,
  title = 'Confirm Deletion',
  message = 'This action cannot be undone.',
  confirmButtonText = 'Confirm',
  cancelButtonText = 'Cancel',
  confirmButtonDisabled = false,
  cancelButtonDisabled = false,
  confirmButtonColor = 'primary',
}) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          borderRadius: 2,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          backdropFilter: 'blur(8px)',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
        },
      }}
      maxWidth="xs"
      fullWidth
    >
      <DialogTitle component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, p: 2 }}>
        <WarningIcon color="error" sx={{ fontSize: 24 }} />
        <Typography variant="h6" component="div" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
          {title}
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
          {message}
        </Typography>
      </DialogContent>
      <DialogActions sx={{ p: 3 }}>
        <Button
          variant="outlined"
          color="inherit"
          onClick={onClose}
          disabled={cancelButtonDisabled}
          sx={{
            textTransform: 'none',
            borderRadius: 2,
            mr: 2,
            '&:hover': {
              backgroundColor: 'rgba(0, 0, 0, 0.04)',
            },
            '&.Mui-disabled': {
              opacity: 0.7,
            },
          }}
        >
          {cancelButtonText}
        </Button>
        <Button
          variant="contained"
          color={confirmButtonColor}
          onClick={onConfirm}
          disabled={confirmButtonDisabled}
          sx={{
            textTransform: 'none',
            borderRadius: 2,
            '&:hover': {
              backgroundColor: `${confirmButtonColor}.dark`,
            },
            '&.Mui-disabled': {
              backgroundColor: `${confirmButtonColor}.light`,
              color: 'text.disabled',
            },
          }}
        >
          {confirmButtonText}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfirmationDialog;
