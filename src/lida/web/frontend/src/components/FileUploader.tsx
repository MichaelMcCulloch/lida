import React, { useCallback, useState } from 'react';
import { useUploadPipeline } from '../hooks/useUploadPipeline';
import { PipelineProgress } from './PipelineProgress';

interface FileUploaderProps {
  onSuccess: (summary: any) => void;
  onError: (error: string) => void;
}

export const FileUploader: React.FC<FileUploaderProps> = ({ onSuccess, onError }) => {
  const [urlBusy, setUrlBusy] = useState(false);
  const { state, start, reset } = useUploadPipeline({
    onComplete: (data) => {
      // The streaming endpoint wraps the same payload shape as /summarize
      // inside { event: 'complete', data: {...} } so we can hand the body
      // straight to the existing app handler.
      if (data && data.status) {
        onSuccess(data);
      } else {
        onError((data && data.message) || 'Upload failed');
      }
    },
    onError: (msg) => onError(msg),
  });

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files?.length) return;
      const file = e.target.files[0];
      // Reset the <input> so picking the same file twice still re-fires.
      e.target.value = '';
      await start(file);
    },
    [start],
  );

  const handleUrlSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const url = (e.currentTarget.elements.namedItem('url') as HTMLInputElement).value;
    if (!url) return;

    setUrlBusy(true);
    try {
      const res = await fetch('/api/v1/summarize/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
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
      setUrlBusy(false);
    }
  };

  const showPipeline = state.active || state.finished;

  return (
    <div className="file-uploader">
      <h3>Data Upload</h3>
      {!showPipeline && (
        <div className="upload-methods">
          <div className="upload-method">
            <label className="button-like">
              Upload File (CSV, JSON, XLSX, SQLite, .tar, .gz, .tar.gz)
              <input
                type="file"
                onChange={handleFileChange}
                accept=".csv,.json,.xlsx,.db,.sqlite,.sqlite3,.tar,.tgz,.gz"
                hidden
              />
            </label>
          </div>
          <div className="divider">OR</div>
          <form onSubmit={handleUrlSubmit} className="upload-method url-form">
            <input type="url" name="url" placeholder="Enter CSV URL..." required />
            <button type="submit" disabled={urlBusy}>
              {urlBusy ? 'Processing...' : 'Load from URL'}
            </button>
          </form>
        </div>
      )}

      {showPipeline && (
        <>
          <PipelineProgress state={state} />
          {state.finished && (
            <div className="pipeline-progress__actions">
              <button type="button" className="secondary" onClick={reset}>
                Upload another file
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};
