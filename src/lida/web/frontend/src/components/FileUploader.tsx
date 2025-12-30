import React, { useState } from 'react';
import pako from 'pako';

interface FileUploaderProps {
  onSuccess: (summary: any) => void;
  onError: (error: string) => void;
}

export const FileUploader: React.FC<FileUploaderProps> = ({ onSuccess, onError }) => {
  const [loading, setLoading] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;

    const file = e.target.files[0];
    setLoading(true);

    try {
      // Read file as ArrayBuffer
      const arrayBuffer = await file.arrayBuffer();
      
      // Compress using pako with maximum compression
      const compressed = pako.gzip(new Uint8Array(arrayBuffer), { level: 9 });
      
      // Create blob from compressed data
      const compressedBlob = new Blob([compressed], { type: 'application/gzip' });
      
      const formData = new FormData();
      // Append with .gz extension so backend knows it's compressed
      formData.append('file', compressedBlob, `${file.name}.gz`);

      const res = await fetch('/api/summarize', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      if (data.status) {
        onSuccess(data);
      } else {
        onError(data.message || 'Upload failed');
      }
    } catch (err: any) {
      onError(err.message || 'Error processing file');
    } finally {
      setLoading(false);
    }
  };

  const handleUrlSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const url = (e.currentTarget.elements.namedItem('url') as HTMLInputElement).value;
    if (!url) return;

    setLoading(true);
    try {
        const res = await fetch('/api/summarize/url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });
        const data = await res.json();
        
        if (data.status) {
            onSuccess(data);
        } else {
            onError(data.message || 'URL processing failed');
        }
    } catch (err: any) {
        onError(err.message || 'Network error');
    } finally {
        setLoading(false);
    }
  }

  return (
    <div className="file-uploader">
      <h3>Data Upload</h3>
      <div className="upload-methods">
        <div className="upload-method">
            <label className="button-like">
                Upload File (CSV, JSON)
                <input type="file" onChange={handleFileChange} accept=".csv,.json,.xlsx" hidden />
            </label>
        </div>
        <div className="divider">OR</div>
        <form onSubmit={handleUrlSubmit} className="upload-method url-form">
            <input type="url" name="url" placeholder="Enter CSV URL..." required />
            <button type="submit" disabled={loading}>
                {loading ? 'Processing...' : 'Load from URL'}
            </button>
        </form>
      </div>
      {loading && <div className="loader">Analyzing data...</div>}
    </div>
  );
};
