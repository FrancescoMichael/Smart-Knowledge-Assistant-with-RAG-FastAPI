export interface QueryRequest {
    query: string;
}

export interface RetrievedChunk {
    index: number;
    score: number;
    preview: string;
}

export interface QueryResponse {
    query: string;
    answer: string;
    retrieved_chunks: RetrievedChunk[];
}

export interface HealthResponse {
    status: string;
    chunks_indexed: number;
    current_pdf?: string;
    has_embeddings: boolean;
}

export interface UploadResponse {
    message: string;
    filename: string;
    chunks_count: number;
    text_length: number;
}

export interface StreamMetadata {
    type: 'metadata';
    query: string;
    retrieved_chunks: RetrievedChunk[];
}

export interface StreamChunk {
    type: 'chunk';
    content: string;
    is_final: boolean;
}

export interface StreamComplete {
    type: 'complete';
    full_answer: string;
}

export interface ChatMessage {
    id: string;
    type: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    chunks?: RetrievedChunk[];
}