import React, { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface ChatInputProps {
    onSendMessage: (message: string) => void;
    isLoading: boolean;
    placeholder?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
    onSendMessage,
    isLoading,
    placeholder = "Ask a question about the document..."
}) => {
    const [input, setInput] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim() && !isLoading) {
            onSendMessage(input.trim());
            setInput('');
        }
    };

    return (
        <form onSubmit={handleSubmit} className="bg-white border-t px-4 py-3 sticky bottom-0 shadow-sm">
            <div className="flex gap-2 max-w-4xl mx-auto">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={placeholder}
                    disabled={isLoading}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                />
                <button
                    type="submit"
                    disabled={!input.trim() || isLoading}
                    className="px-4 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 disabled:opacity-50 flex items-center gap-2"
                >
                    {isLoading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                        <Send className="w-4 h-4" />
                    )}
                </button>
            </div>
        </form>
    );
};