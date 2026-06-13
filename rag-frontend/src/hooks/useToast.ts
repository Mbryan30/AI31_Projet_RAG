import { useCallback } from 'react'
import { useAppDispatch } from './redux'
import { showToast, hideToast } from '@/store/slices/uiSlice'

export function useToast() {
  const dispatch = useAppDispatch()

  const toast = useCallback(
    (icon: string, message: string, duration = 2600) => {
      dispatch(showToast({ icon, message }))
      setTimeout(() => dispatch(hideToast()), duration)
    },
    [dispatch],
  )

  return { toast }
}
