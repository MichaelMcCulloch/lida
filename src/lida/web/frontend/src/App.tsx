import { useCallback, useMemo, useState } from 'react';
import './App.css';
import { ModelSelector } from './components/ModelSelector';
import { FileUploader } from './components/FileUploader';
import { GoalsBoard } from './components/GoalsBoard';
import { Visualizer } from './components/Visualizer';
import { useUploadPipeline } from './hooks/useUploadPipeline';
import type { GoalEntry, TableProgress } from './hooks/useUploadPipeline';

interface SelectedGoal {
  tableName: string;
  goal: GoalEntry;
}

function App() {
  const [activeModel, setActiveModel] = useState<{ provider: string; model: string }>({ provider: '', model: '' });
  const [nGoals, setNGoals] = useState(5);
  const [activeTableName, setActiveTableName] = useState<string | null>(null);
  const [selectedGoal, setSelectedGoal] = useState<SelectedGoal | null>(null);
  const [errorBanner, setErrorBanner] = useState<string>('');
  const [urlResult, setUrlResult] = useState<{
    summary: any;
    goals: any[];
    charts: any[];
    data_filename?: string;
  } | null>(null);

  const pipeline = useUploadPipeline({
    onComplete: () => {
      setErrorBanner('');
    },
    onError: (msg) => setErrorBanner(msg),
  });

  const handleModelChange = useCallback((provider: string, model: string) => {
    setActiveModel({ provider, model });
  }, []);

  const handleStart = useCallback(
    (file: File) => {
      setErrorBanner('');
      setSelectedGoal(null);
      setActiveTableName(null);
      setUrlResult(null);
      pipeline.start(file, { nGoals });
    },
    [pipeline, nGoals],
  );

  const handleReset = useCallback(() => {
    setSelectedGoal(null);
    setActiveTableName(null);
    setErrorBanner('');
    setUrlResult(null);
    pipeline.reset();
  }, [pipeline]);

  const handleUrlSubmit = useCallback(
    async (url: string) => {
      setErrorBanner('');
      setSelectedGoal(null);
      setActiveTableName(null);
      try {
        const res = await fetch('/api/v1/summarize/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
        const data = await res.json();
        if (!data.status) {
          setErrorBanner(data.message || 'URL processing failed');
          return;
        }
        setUrlResult({
          summary: data.summary,
          goals: data.goals || [],
          charts: data.charts || [],
          data_filename: data.data_filename,
        });
      } catch (err: any) {
        setErrorBanner(err?.message || 'Network error');
      }
    },
    [],
  );

  const tables = pipeline.state.tables;
  const isMultiTable = tables.length > 1 || pipeline.state.dispatchKind === 'sqlite' || pipeline.state.dispatchKind === 'tar';

  const activeTable: TableProgress | null = useMemo(() => {
    if (tables.length === 0) return null;
    if (activeTableName) {
      const found = tables.find((t) => t.name === activeTableName);
      if (found) return found;
    }
    return tables[0];
  }, [tables, activeTableName]);

  // The progressive view appears the moment the first summary lands, even
  // before the pipeline finishes — so the user sees goals/charts streaming in.
  const hasProgressiveSummary = !!activeTable?.summary;
  const showResults = hasProgressiveSummary || urlResult !== null;

  // For URL upload (non-streaming), build a synthetic "table" view so the
  // same GoalsBoard layout works.
  const urlTable: TableProgress | null = useMemo(() => {
    if (!urlResult) return null;
    const charts: Record<number, any> = {};
    const plotStatuses: Record<number, any> = {};
    (urlResult.charts || []).forEach((chart: any, i: number) => {
      charts[i] = chart;
      plotStatuses[i] = 'rendered';
    });
    return {
      name: urlResult.data_filename || 'remote dataset',
      status: 'done',
      chartsRendered: (urlResult.charts || []).length,
      chartsTotal: (urlResult.goals || []).length,
      summary: urlResult.summary,
      goals: (urlResult.goals || []).map((g: any, i: number) => ({
        index: typeof g.index === 'number' ? g.index : i,
        question: g.question || '',
        visualization: g.visualization || '',
        rationale: g.rationale || '',
      })),
      charts,
      plotStatuses,
      plotErrors: {},
    };
  }, [urlResult]);

  const displayedTable: TableProgress | null = activeTable ?? urlTable;
  const summary = displayedTable?.summary;

  return (
    <div className="container">
      <header className="header">
        <h1>LIDA</h1>
        <p>Automatic Generation of Visualizations and Infographics using Large Language Models</p>
      </header>

      <main style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '2rem', alignItems: 'center' }}>
        <section className="controls" style={{ maxWidth: '800px' }}>
          <ModelSelector onModelChange={handleModelChange} />
          <div className="status" style={{ marginTop: '1rem', color: 'var(--color-text-muted)' }}>
            Active Model: <span className="highlight">{activeModel.model || 'None'}</span> <small>({activeModel.provider})</small>
          </div>
        </section>

        {errorBanner && (
          <div
            className="error-banner"
            style={{
              backgroundColor: '#ef444422',
              border: '1px solid #ef4444',
              color: '#fca5a5',
              padding: '1rem',
              borderRadius: '8px',
              width: '100%',
              maxWidth: '800px',
            }}
          >
            {errorBanner}
          </div>
        )}

        <section className="upload-area">
          <FileUploader
            nGoals={nGoals}
            onNGoalsChange={setNGoals}
            onStart={handleStart}
            onUrlSubmit={handleUrlSubmit}
            onReset={handleReset}
            pipeline={pipeline.state}
          />
        </section>

        {showResults && summary && displayedTable && (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {isMultiTable && tables.length > 0 && (
              <section className="tables-area">
                <div className="summary-card">
                  <h3>Tables</h3>
                  <p style={{ color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                    {tables.length} table{tables.length === 1 ? '' : 's'} detected. Select one to explore.
                  </p>
                  <div style={{ marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {tables.map((t) => (
                      <button
                        key={t.name}
                        onClick={() => {
                          setActiveTableName(t.name);
                          setSelectedGoal(null);
                        }}
                        className={displayedTable?.name === t.name ? 'primary' : 'secondary'}
                        style={{ fontSize: '0.9rem' }}
                      >
                        {t.name}
                        <small style={{ marginLeft: '0.4rem', opacity: 0.75 }}>
                          ({t.goals.length}/{t.chartsTotal || nGoals})
                        </small>
                      </button>
                    ))}
                  </div>
                </div>
              </section>
            )}

            <section className="summary-area">
              <div className="summary-card">
                <h3>Dataset Summary{displayedTable.name ? ` — ${displayedTable.name}` : ''}</h3>
                <div style={{ marginTop: '1rem', display: 'grid', gap: '0.5rem' }}>
                  <div><strong>Name:</strong> {summary.name}</div>
                  <div><strong>Description:</strong> {summary.dataset_description}</div>
                </div>
              </div>
            </section>

            <section className="goals-area">
              <h3>Goals & Visualizations</h3>
              <GoalsBoard
                goals={displayedTable.goals}
                charts={displayedTable.charts}
                plotStatuses={displayedTable.plotStatuses}
                plotErrors={displayedTable.plotErrors}
                totalExpected={displayedTable.chartsTotal || nGoals}
                onGoalSelect={(goal) => setSelectedGoal({ tableName: displayedTable.name, goal })}
                selectedIndex={selectedGoal?.tableName === displayedTable.name ? selectedGoal.goal.index : null}
              />
            </section>

            {selectedGoal && selectedGoal.tableName === displayedTable.name && (
              <section className="visualization-area">
                <h3>Visualization</h3>
                <p style={{ marginBottom: '1rem' }}>
                  <strong style={{ color: 'var(--color-accent)' }}>Goal:</strong> {selectedGoal.goal.question}
                </p>
                <Visualizer
                  summary={summary}
                  goal={selectedGoal.goal}
                  library="seaborn"
                />
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
