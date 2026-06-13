import { configureStore } from '@reduxjs/toolkit'
import sessionsReducer from './slices/sessionsSlice'
import uiReducer       from './slices/uiSlice'

export const store = configureStore({
  reducer: {
    sessions: sessionsReducer,
    ui:       uiReducer,
  },
})

export type RootState   = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
