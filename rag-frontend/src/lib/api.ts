import axios from 'axios'
import type { QueryRequest, QueryResponse } from '@/types'

const client = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 300_000,
  headers: { 'Content-Type': 'application/json' },
})

// Attach auth token if present (JWT / API key)
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('rag_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// ─── RAG query (non-streaming) ────────────────────────────────
export async function sendQuery(payload: QueryRequest): Promise<QueryResponse> {

  const { data } = await client.post<QueryResponse>('/query', payload)
  console.log(client.head)
  return data
}

// ─── RAG query (SSE streaming) ────────────────────────────────
// Returns an EventSource you can listen to; caller is responsible for closing it.
export function streamQuery(
  payload: QueryRequest,
  onToken: (token: string) => void,
  onDone: (meta: { sources: QueryResponse['sources']; metrics: QueryResponse['metrics'] }) => void,
  onError: (err: Error) => void,
): () => void {
  const params = new URLSearchParams({
    question:   payload.question,
    session_id: payload.session_id,
    mode:       payload.mode,
    sources:    payload.sources.join(','),
  })
  const url = `/api/stream?${params}`
  const es  = new EventSource(url)

  es.addEventListener('token', (e) => onToken((e as MessageEvent).data))
  es.addEventListener('done',  (e) => {
    onDone(JSON.parse((e as MessageEvent).data))
    es.close()
  })
  es.onerror = () => {
    onError(new Error('Stream error'))
    es.close()
  }

  return () => es.close()
}

export default client
