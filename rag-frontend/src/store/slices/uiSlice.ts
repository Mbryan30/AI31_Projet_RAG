import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { RootState } from '../index'

interface Toast { icon: string; message: string }
interface UIState {
  sidebarOpen:   boolean
  renameModal:   { open: boolean; sessionId: string | null }
  toast:         Toast & { visible: boolean }
  activeSources: string[]
}

const initialState: UIState = {
  sidebarOpen:  true,
  renameModal:  { open: false, sessionId: null },
  toast:        { icon: '', message: '', visible: false },
  activeSources: ['vectorstore'],
}

export const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar(state)                        { state.sidebarOpen = !state.sidebarOpen },
    setSidebarOpen(state, a: PayloadAction<boolean>) { state.sidebarOpen = a.payload },

    openRenameModal(state, a: PayloadAction<string>) {
      state.renameModal = { open: true, sessionId: a.payload }
    },
    closeRenameModal(state) {
      state.renameModal = { open: false, sessionId: null }
    },

    showToast(state, a: PayloadAction<Toast>) {
      state.toast = { ...a.payload, visible: true }
    },
    hideToast(state) { state.toast.visible = false },

    toggleSource(state, a: PayloadAction<string>) {
      const src = a.payload
      state.activeSources = state.activeSources.includes(src)
        ? state.activeSources.filter((s) => s !== src)
        : [...state.activeSources, src]
    },
  },
})

export const {
  toggleSidebar, setSidebarOpen,
  openRenameModal, closeRenameModal,
  showToast, hideToast,
  toggleSource,
} = uiSlice.actions

export default uiSlice.reducer

export const selectSidebarOpen   = (s: RootState) => s.ui.sidebarOpen
export const selectRenameModal   = (s: RootState) => s.ui.renameModal
export const selectToast         = (s: RootState) => s.ui.toast
export const selectActiveSources = (s: RootState) => s.ui.activeSources
