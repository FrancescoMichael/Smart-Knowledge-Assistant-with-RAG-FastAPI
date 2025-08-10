import React from 'react';
import { ChatMessage } from '../types/api';
import { User, Bot, Clock, FileText, AlertCircle, CheckCircle } from 'lucide-react';

interface MessageBubbleProps {
    message: ChatMessage;
    showChunks?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
    message,
    showChunks = false
}) => {
    const isUser = message.type === 'user';
    const isSystem = message.type === 'system';

    const getMessageStyle = () => {
        if (isSystem) {
            return 'bg-blue-50 text-blue-800 border border-blue-200';
        }
        if (isUser) {
            return 'bg-blue-500 text-white rounded-br-sm';
        }
        return 'bg-gray-100 text-gray-900 rounded-bl-sm';
    };

    const getIcon = () => {
        if (isSystem) return <AlertCircle className="w-4 h-4 text-blue-500" />;
        if (isUser) return <User className="w-4 h-4 text-white" />;
        return <Bot className="w-4 h-4 text-white" />;
    };

    const getAvatarStyle = () => {
        if (isSystem) return 'bg-blue-100';
        if (isUser) return 'bg-blue-500';
        return 'bg-gray-600';
    };

    if (isSystem) {
        return (
            <div className="flex justify-center mb-4">
                <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg text-blue-800 text-sm">
                    <CheckCircle className="w-4 h-4" />
                    {message.content}
                </div>
            </div>
        );
    }

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 shadow-sm ${isUser
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-gray-100 text-gray-900 rounded-bl-none'
                    }`}
            >
                <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
        </div>
    );
};