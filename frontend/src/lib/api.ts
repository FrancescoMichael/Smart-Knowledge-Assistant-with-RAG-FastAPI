import { HealthResponse, QueryResponse, StreamMetadata, UploadResponse } from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export class ApiService {
    static async checkHealth(): Promise<HealthResponse> {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('API health check failed');
        }
        return response.json();
    }

    static async uploadFile(file: File): Promise<UploadResponse> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to upload file');
        }

        return response.json();
    }

    static async clearDocument(): Promise<void> {
        const response = await fetch(`${API_BASE_URL}/document`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to clear document');
        }
    }

    static async askQuestion(query: string): Promise<QueryResponse> {
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get answer');
        }

        return response.json();
    }

    static async askQuestionStream(
        query: string,
        onChunk: (chunk: string, isFinal?: boolean) => void,
        onMetadata?: (metadata: StreamMetadata) => void,
        onComplete?: (fullAnswer: string) => void
    ): Promise<void> {
        const response = await fetch(`${API_BASE_URL}/ask_stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error('Failed to get streaming response');
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.type === 'metadata') {
                                onMetadata?.(data);
                            } else if (data.type === 'chunk') {
                                onChunk?.(data.content, data.is_final);
                            } else if (data.type === 'complete') {
                                onComplete?.(data.full_answer);
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE data:', line);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }
}