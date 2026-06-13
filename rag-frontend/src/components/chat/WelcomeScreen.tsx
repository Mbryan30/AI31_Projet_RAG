import { useChat } from '@/hooks/useChat'

const SUGGESTIONS = [
  { title: 'Droits des personnes',   sub: 'Droit d\'accès, effacement, portabilité' },
  { title: 'Bases légales',           sub: 'Consentement, intérêt légitime, contrat' },
  { title: 'DPO & registre',          sub: 'Obligations du délégué à la protection' },
  { title: 'Transferts hors UE',      sub: 'Clauses contractuelles types, BCR' },
]

export function WelcomeScreen() {
  const { send } = useChat()

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-6 gap-4 animate-fadeUp">
      <div className="w-14 h-14 bg-bg-2 border border-border-md rounded-lg flex items-center justify-center text-2xl mb-1">⬡</div>
      <h1 className="font-serif text-[28px] text-tx font-normal">RAG·AI — RGPD</h1>
      <p className="text-[14px] text-tx-2 max-w-md leading-relaxed">
        Pipeline RAG adaptatif avec CRAG, Self-RAG et indexation hiérarchique RAPTOR.
        Posez vos questions sur la conformité RGPD.
      </p>
      <div className="grid grid-cols-2 gap-2.5 max-w-[480px] w-full mt-2">
        {SUGGESTIONS.map(({ title, sub }) => (
          <button
            key={title}
            onClick={() => send(`${title} — ${sub}`)}
            className="bg-bg-1 border border-border rounded-sm px-3.5 py-3 text-left hover:border-border-md hover:bg-bg-2 transition-all"
          >
            <div className="text-[12.5px] font-medium text-tx mb-0.5">{title}</div>
            <div className="text-[11px] text-tx-3">{sub}</div>
          </button>
        ))}
      </div>
    </div>
  )
}
