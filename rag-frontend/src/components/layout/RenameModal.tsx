import { useState, useEffect } from 'react'
import { useAppDispatch, useAppSelector } from '@/hooks/redux'
import { renameSession, selectSessions } from '@/store/slices/sessionsSlice'
import { closeRenameModal, selectRenameModal } from '@/store/slices/uiSlice'
import { useToast } from '@/hooks/useToast'
import { Modal } from '@/components/ui/Modal'
import { Button } from '@/components/ui/Button'

export function RenameModal() {
  const dispatch  = useAppDispatch()
  const modal     = useAppSelector(selectRenameModal)
  const sessions  = useAppSelector(selectSessions)
  const { toast } = useToast()
  const [value, setValue] = useState('')

  useEffect(() => {
    if (modal.open && modal.sessionId) {
      const s = sessions.find((x) => x.id === modal.sessionId)
      setValue(s?.title ?? '')
    }
  }, [modal, sessions])

  function handleConfirm() {
    if (!value.trim() || !modal.sessionId) return
    dispatch(renameSession({ id: modal.sessionId, title: value.trim() }))
    dispatch(closeRenameModal())
    toast('✎', 'Session renommée')
  }

  return (
    <Modal
      open={modal.open}
      onClose={() => dispatch(closeRenameModal())}
      title="Renommer la session"
    >
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleConfirm()}
        placeholder="Titre de la session…"
        autoFocus
        className="w-full bg-bg-2 border border-border-md rounded-sm px-3 py-2.5 text-[14px] text-tx placeholder-tx-3 outline-none focus:border-accent/50 transition-colors mb-4"
      />
      <div className="flex justify-end gap-2">
        <Button onClick={() => dispatch(closeRenameModal())}>Annuler</Button>
        <Button variant="primary" onClick={handleConfirm}>Renommer</Button>
      </div>
    </Modal>
  )
}
