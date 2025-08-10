import React from 'react';
import { CheckCircle, XCircle, AlertCircle, FileText } from 'lucide-react';

interface StatusBannerProps {
  status: 'connected' | 'error' | 'loading';
  chunksIndexed?: number;
  currentPdf?: string;
  error?: string;
}

export const StatusBanner: React.FC<StatusBannerProps> = ({
  status,
  chunksIndexed,
  currentPdf,
  error
}) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        const message = currentPdf
          ? `Connected - "${currentPdf}" (${chunksIndexed} chunks)`
          : `Connected - ${chunksIndexed} chunks indexed`;
        return {
          icon: CheckCircle,
          bgColor: 'bg-green-50',
          textColor: 'text-green-800',
          iconColor: 'text-green-500',
          message
        };
      case 'error':
        return {
          icon: XCircle,
          bgColor: 'bg-red-50',
          textColor: 'text-red-800',
          iconColor: 'text-red-500',
          message: error || 'Connection error'
        };
      case 'loading':
        return {
          icon: AlertCircle,
          bgColor: 'bg-yellow-50',
          textColor: 'text-yellow-800',
          iconColor: 'text-yellow-500',
          message: 'Connecting to backend...'
        };
    }
  };

  const { icon: Icon, bgColor, textColor, iconColor, message } = getStatusConfig();

  return (
    <div className={`${bgColor} ${textColor} px-4 py-2 border-b`}>
      <div className="flex items-center gap-2">
        <Icon className={`w-4 h-4 ${iconColor}`} />
        <span className="text-sm font-medium">{message}</span>
        {currentPdf && (
          <div className="ml-2 flex items-center gap-1 text-xs">
            <FileText className="w-3 h-3" />
            <span>Document loaded</span>
          </div>
        )}
      </div>
    </div>
  );
};