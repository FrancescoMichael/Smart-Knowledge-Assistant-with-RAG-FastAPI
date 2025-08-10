import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { ApiService } from '../lib/api';
import type { RetrievedChunk, StreamMetadata } from '../types/api';

type ApiStatus = 'loading' | 'connected' | 'error';

export default function Home() {
    const [question, setQuestion] = useState<string>('');
    const [answer, setAnswer] = useState<string>('');
    const [streamingAnswer, setStreamingAnswer] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [isStreaming, setIsStreaming] = useState<boolean>(false);
    const [isUploading, setIsUploading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');
    const [currentPdf, setCurrentPdf] = useState<string>('');
    const [chunksIndexed, setChunksIndexed] = useState<number>(0);
    const [retrievedChunks, setRetrievedChunks] = useState<RetrievedChunk[]>([]);
    const [apiStatus, setApiStatus] = useState<ApiStatus>('loading');
    const [useStreaming, setUseStreaming] = useState<boolean>(false);

    useEffect(() => {
        checkApiHealth();
    }, []);

    const checkApiHealth = async (): Promise<void> => {
        try {
            const health = await ApiService.checkHealth();
            setCurrentPdf(health.current_pdf || 'No PDF loaded');
            setChunksIndexed(health.chunks_indexed || 0);
            setApiStatus('connected');
            setError('');
        } catch (err) {
            setApiStatus('error');
            setError(err instanceof Error ? err.message : 'Unknown error');
        }
    };

    const handleFileUpload = async (file: File | null): Promise<void> => {
        if (!file) return;
        setIsUploading(true);
        try {
            const res = await ApiService.uploadFile(file);
            setCurrentPdf(res.filename);
            setChunksIndexed(res.chunks_count);
            setError('');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setIsUploading(false);
        }
    };

    const clearDocument = async (): Promise<void> => {
        try {
            await ApiService.clearDocument();
            setCurrentPdf('No PDF loaded');
            setChunksIndexed(0);
            setAnswer('');
            setStreamingAnswer('');
            setRetrievedChunks([]);
            setError('');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to clear document');
        }
    };

    const handleAsk = async (): Promise<void> => {
        if (!question.trim()) return;
        setError('');
        setAnswer('');
        setStreamingAnswer('');
        setRetrievedChunks([]);

        if (useStreaming) {
            setIsStreaming(true);
            try {
                await ApiService.askQuestionStream(
                    question,
                    (chunk: string, isFinal?: boolean) => {
                        setStreamingAnswer(prev => prev + chunk);
                        if (isFinal) {
                            setIsStreaming(false);
                        }
                    },
                    (metadata: StreamMetadata) => {
                        setRetrievedChunks(metadata.retrieved_chunks || []);
                    },
                    (fullAnswer: string) => {
                        setStreamingAnswer(fullAnswer);
                        setIsStreaming(false);
                    }
                );
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Streaming failed');
            } finally {
                setIsStreaming(false);
            }
        } else {
            setIsLoading(true);
            try {
                const res = await ApiService.askQuestion(question);
                setAnswer(res.answer);
                setRetrievedChunks(res.retrieved_chunks || []);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Unknown error');
            } finally {
                setIsLoading(false);
            }
        }
    };

    const displayAnswer: string = useStreaming ? streamingAnswer : answer;
    const isProcessing: boolean = isLoading || isStreaming;

    const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
        const file = event.target.files?.[0] || null;
        handleFileUpload(file);
    };

    const handleTextareaChange = (event: React.ChangeEvent<HTMLTextAreaElement>): void => {
        setQuestion(event.target.value);
    };

    const handleStreamingToggle = (event: React.ChangeEvent<HTMLInputElement>): void => {
        setUseStreaming(event.target.checked);
    };

    return (
        <>
            <Head>
                <title>Smart Knowledge Assistant</title>
            </Head>

            <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50 py-12 px-4">
                <div className="max-w-5xl mx-auto space-y-10">

                    <div className="text-center space-y-3">
                        <h1 className="text-5xl font-extrabold text-gray-900 drop-shadow-sm">
                            Smart Knowledge Assistant
                        </h1>
                        <p className="text-gray-600 text-lg">
                            Upload a PDF and ask questions about its content.
                        </p>
                    </div>

                    <div className="flex items-center justify-center gap-4 p-5 bg-white rounded-xl shadow-md border border-gray-100">
                        <button
                            onClick={checkApiHealth}
                            className="px-5 py-2.5 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-lg hover:from-gray-600 hover:to-gray-700 transition-all duration-200 shadow-sm"
                            type="button"
                        >
                            Check API Health
                        </button>
                        <span className={`text-lg font-semibold ${apiStatus === 'connected'
                            ? 'text-green-600'
                            : apiStatus === 'error'
                                ? 'text-red-600'
                                : 'text-yellow-600'
                            }`}>
                            {apiStatus === 'loading' ? 'Checking...' :
                                apiStatus === 'connected' ? 'Connected' : 'Error'}
                        </span>
                    </div>

                    <div className="p-8 bg-white rounded-xl shadow-md border border-gray-100 space-y-6">
                        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                            Document Management
                        </h2>

                        <div className="grid md:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Upload PDF
                                </label>
                                <input
                                    type="file"
                                    accept="application/pdf"
                                    onChange={handleFileInputChange}
                                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200 transition"
                                />
                                {isUploading && (
                                    <p className="text-blue-500 mt-2 text-sm animate-pulse">
                                        Uploading and processing...
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="p-8 bg-white rounded-xl shadow-md border border-gray-100 space-y-5">
                        <h2 className="text-2xl font-bold text-gray-800">Ask a Question</h2>

                        {error && (
                            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                                {error}
                            </div>
                        )}

                        <div className="flex items-center space-x-3">
                            <input
                                type="checkbox"
                                id="streaming"
                                checked={useStreaming}
                                onChange={handleStreamingToggle}
                                className="rounded"
                            />
                            <label htmlFor="streaming" className="text-sm font-medium text-gray-700">
                                Use streaming response
                            </label>
                        </div>

                        <textarea
                            value={question}
                            onChange={handleTextareaChange}
                            placeholder="Type your question here..."
                            className="w-full p-4 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800"
                            rows={3}
                        />

                        <button
                            onClick={handleAsk}
                            disabled={isProcessing || !question.trim()}
                            className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-lg hover:from-blue-600 hover:to-indigo-600 disabled:bg-gray-400 transition-all duration-200 font-medium shadow-sm"
                            type="button"
                        >
                            {isProcessing ? (useStreaming ? 'Streaming...' : 'Asking...') : 'Ask Question'}
                        </button>
                    </div>

                    {displayAnswer && (
                        <div className="p-8 bg-white rounded-xl shadow-md border border-gray-100">
                            <div className="flex items-center justify-between mb-4">
                                <h2 className="text-2xl font-bold text-gray-800">Answer</h2>
                            </div>
                            <div className="prose max-w-none">
                                <p className="text-gray-800 leading-relaxed">{displayAnswer}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}
