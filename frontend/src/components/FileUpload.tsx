import React, { useCallback, useState } from 'react';
import { Upload, X, FileText, Loader2, Trash2 } from 'lucide-react';

interface FileUploadProps {
    onFileUpload: (file: File) => void;
    onClearDocument: () => void;
    isUploading: boolean;
    currentDocument?: string;
    chunksCount?: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({
    onFileUpload,
    onClearDocument,
    isUploading,
    currentDocument,
    chunksCount
}) => {
    const [dragActive, setDragActive] = useState(false);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
                onFileUpload(file);
            } else {
                alert('Please upload a PDF file');
            }
        }
    }, [onFileUpload]);

    const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            onFileUpload(file);
        }
    }, [onFileUpload]);

    if (currentDocument) {
        return (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-green-100 rounded-full">
                            <FileText className="w-5 h-5 text-green-600" />
                        </div>
                        <div>
                            <div className="font-medium text-green-900">{currentDocument}</div>
                            <div className="text-sm text-green-700">{chunksCount} chunks indexed</div>
                        </div>
                    </div>
                    <button
                        onClick={onClearDocument}
                        className="p-2 text-red-500 hover:bg-red-50 rounded-full transition-colors"
                        title="Clear document"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="mb-4">
            <div
                className={`
          relative border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${dragActive
                        ? 'border-blue-400 bg-blue-50'
                        : 'border-gray-300 bg-gray-50 hover:bg-gray-100'
                    }
          ${isUploading ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
        `}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    accept=".pdf"
                    onChange={handleChange}
                    disabled={isUploading}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                />

                <div className="flex flex-col items-center gap-3">
                    {isUploading ? (
                        <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                    ) : (
                        <Upload className="w-8 h-8 text-gray-400" />
                    )}

                    <div>
                        <p className="text-lg font-medium text-gray-900">
                            {isUploading ? 'Processing PDF...' : 'Upload a PDF document'}
                        </p>
                        <p className="text-sm text-gray-500 mt-1">
                            {isUploading
                                ? 'Please wait while we extract and index the content'
                                : 'Drag and drop your PDF file here, or click to browse'
                            }
                        </p>
                    </div>

                    {!isUploading && (
                        <div className="text-xs text-gray-400">
                            Maximum file size: 50MB • PDF files only
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};