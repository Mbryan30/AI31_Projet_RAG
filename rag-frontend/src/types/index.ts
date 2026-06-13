// ─── Core domain types ───────────────────────────────────────

export type MessageRole = 'user' | 'assistant'

export interface SourceRef {
  title: string
  score: number       // 0–100
  url?: string
}

export interface MessageMetrics {
  latency: number     // ms
  tokens: number
  strategy: string    // e.g. "CRAG", "HyDE"
}

export interface Message {
  id: string
  role: MessageRole
  content: string
  ts: number          // unix ms
  sources?: SourceRef[]
  metrics?: MessageMetrics
  isStreaming?: boolean
}

export interface Session {
  id: string
  title: string
  messages: Message[]
  created: number
  updated: number
  mode: RagMode
}

// ─── RAG modes ───────────────────────────────────────────────

export type RagMode = 'adaptive' | 'crag' | 'hyde' | 'selfrag'

export interface RagModeConfig {
  id: RagMode
  label: string
  description: string
  color: string
}

export const RAG_MODES: RagModeConfig[] = [
  { id: 'adaptive', label: 'RAG Adaptatif', description: 'Pipeline complet avec routing intelligent',  color: '#6c8fff' },
  { id: 'crag',     label: 'CRAG + Fusion', description: 'Corrective RAG avec Reciprocal Rank Fusion', color: '#3ddc84' },
  { id: 'hyde',     label: 'HyDE + Rerank', description: 'Hypothetical Documents + Cross-encoder',      color: '#a855f7' },
  { id: 'selfrag',  label: 'Self-RAG',      description: 'Auto-évaluation ISREL/ISSUP/ISUSE',           color: '#f5a623' },
]

// ─── Data sources ─────────────────────────────────────────────

export type SourceId = 'vectorstore' | 'postgres' | 'neo4j' | 'web'

export interface DataSource {
  id: SourceId
  label: string
  color: string
}

export const DATA_SOURCES: DataSource[] = [
  { id: 'vectorstore', label: 'VectorStore', color: '#6c8fff' },
  { id: 'postgres',    label: 'PostgreSQL',  color: '#3ddc84' },
  { id: 'neo4j',       label: 'Neo4j Graph', color: '#a855f7' },
  { id: 'web',         label: 'Web (CRAG)',  color: '#f5a623' },
]

// ─── API ─────────────────────────────────────────────────────

export interface QueryRequest {
  question: string
  session_id: string
  mode: RagMode
  sources: SourceId[]
}

export interface QueryResponse {
  answer: string
  sources: SourceRef[]
  metrics: MessageMetrics
}
