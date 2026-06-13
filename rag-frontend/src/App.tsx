import { Sidebar }     from '@/components/sidebar/Sidebar'
import { TopBar }      from '@/components/layout/TopBar'
import { ChatArea }    from '@/components/chat/ChatArea'
import { ChatInput }   from '@/components/chat/ChatInput'
import { RenameModal } from '@/components/layout/RenameModal'
import { Toast }       from '@/components/ui/Toast'

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden bg-bg text-tx font-sans antialiased">
      <Sidebar />
      <main className="flex flex-col flex-1 min-w-0">
        <TopBar />
        <ChatArea />
        <ChatInput />
      </main>
      <RenameModal />
      <Toast />
    </div>
  )
}
