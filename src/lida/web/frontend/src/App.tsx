import { useState, useCallback } from 'react'
import './App.css'
import { ModelSelector } from './components/ModelSelector'

import { FileUploader } from './components/FileUploader';

interface Summary {
    name: string;
    file_name: string;
    dataset_description: string;
    field_names: string[];
    fields?: any[];
}

interface TableResult {
    table_name: string;
    data_filename: string;
    summary: Summary;
    goals?: any[];
    charts?: any[];
}

import { GoalGenerator } from './components/GoalGenerator';
import { Visualizer } from './components/Visualizer';

function App() {
  const [activeModel, setActiveModel] = useState<{provider: string, model: string}>({provider: '', model: ''});
  const [summary, setSummary] = useState<Summary | null>(null);
  const [selectedGoal, setSelectedGoal] = useState<any>(null);
  const [initialGoals, setInitialGoals] = useState<any[]>([]);
  const [initialCharts, setInitialCharts] = useState<any[]>([]);
  const [error, setError] = useState<string>('');
  const [tables, setTables] = useState<TableResult[] | null>(null);
  const [activeTable, setActiveTable] = useState<string | null>(null);
  const [databaseFilename, setDatabaseFilename] = useState<string | null>(null);

  const handleModelChange = useCallback((provider: string, model: string) => {
    setActiveModel({ provider, model });
    console.log(`Selected: ${provider} - ${model}`);
  }, []);

  const applyTable = (t: TableResult) => {
      setSummary(t.summary);
      setInitialGoals(t.goals || []);
      setInitialCharts(t.charts || []);
      setActiveTable(t.table_name);
      setSelectedGoal(null);
  };

  const handleUploadSuccess = (data: any) => {
      setError('');
      setSelectedGoal(null);
      if (data.is_database && Array.isArray(data.tables) && data.tables.length > 0) {
          setTables(data.tables);
          setDatabaseFilename(data.data_filename || null);
          applyTable(data.tables[0]);
      } else {
          setTables(null);
          setDatabaseFilename(null);
          setActiveTable(null);
          setSummary(data.summary);
          setInitialGoals(data.goals || []);
          setInitialCharts(data.charts || []);
      }
  };

  const handleUploadError = (msg: string) => {
      setError(msg);
      setSummary(null);
      setInitialGoals([]);
      setInitialCharts([]);
      setSelectedGoal(null);
      setTables(null);
      setActiveTable(null);
      setDatabaseFilename(null);
  };

  const handleGoalSelect = (goal: any) => {
      setSelectedGoal(goal);
      // Scroll to viz area
      setTimeout(() => {
          document.querySelector('.visualization-area')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
  };

  return (
    <div className="container">
      <header className="header">
        <h1>LIDA</h1>
        <p>Automatic Generation of Visualizations and Infographics using Large Language Models</p>
      </header>
      
      <main style={{width: '100%', display: 'flex', flexDirection: 'column', gap: '2rem', alignItems: 'center'}}>
        <section className="controls" style={{maxWidth: '800px'}}>
            <ModelSelector onModelChange={handleModelChange} />
            <div className="status" style={{marginTop: '1rem', color: 'var(--color-text-muted)'}}>
                Active Model: <span className="highlight">{activeModel.model || 'None'}</span> <small>({activeModel.provider})</small>
            </div>
        </section>

        {error && <div className="error-banner" style={{backgroundColor: '#ef444422', border: '1px solid #ef4444', color: '#fca5a5', padding: '1rem', borderRadius: '8px', width: '100%', maxWidth: '800px'}}>{error}</div>}

        {!summary ? (
            <section className="upload-area">
                <FileUploader onSuccess={handleUploadSuccess} onError={handleUploadError} />
            </section>
        ) : (
            <div style={{width: '100%', display: 'flex', flexDirection: 'column', gap: '2rem'}}>
                {tables && (
                    <section className="tables-area">
                        <div className="summary-card">
                            <h3>Database Tables</h3>
                            <p style={{color: 'var(--color-text-muted)', marginTop: '0.25rem'}}>
                                <strong>{databaseFilename}</strong> — {tables.length} table{tables.length === 1 ? '' : 's'}. Select a table to explore.
                            </p>
                            <div style={{marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem'}}>
                                {tables.map((t) => (
                                    <button
                                        key={t.table_name}
                                        onClick={() => applyTable(t)}
                                        className={activeTable === t.table_name ? 'primary' : 'secondary'}
                                        style={{fontSize: '0.9rem'}}
                                    >
                                        {t.table_name}
                                        <small style={{marginLeft: '0.4rem', opacity: 0.75}}>
                                            ({t.summary.field_names?.length ?? 0} cols)
                                        </small>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </section>
                )}

                <section className="summary-area">
                    <div className="summary-card">
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                            <h3>Dataset Summary{activeTable ? ` — ${activeTable}` : ''}</h3>
                            <button className="secondary" onClick={() => { setSummary(null); setSelectedGoal(null); setTables(null); setActiveTable(null); setDatabaseFilename(null); }} style={{fontSize: '0.9rem'}}>Reset / New Upload</button>
                        </div>
                        <div style={{marginTop: '1rem', display: 'grid', gap: '0.5rem'}}>
                            <div><strong>Name:</strong> {summary.name}</div>
                            <div><strong>Description:</strong> {summary.dataset_description}</div>
                        </div>
                    </div>
                </section>

                <section className="goals-area">
                    <h3>Goal Generation</h3>
                    <GoalGenerator summary={summary} onGoalSelect={handleGoalSelect} initialGoals={initialGoals} />
                </section>

                {initialCharts.length > 0 && (
                     <section className="visualization-area">
                        <h3>Generated Visualizations</h3>
                        <Visualizer summary={summary} goal={null} library="seaborn" precomputedCharts={initialCharts} />
                    </section>
                )}

                {selectedGoal && (
                    <section className="visualization-area">
                        <h3>Visualization</h3>
                        <p style={{marginBottom: '1rem'}}><strong style={{color: 'var(--color-accent)'}}>Goal:</strong> {selectedGoal.question}</p>
                        <Visualizer summary={summary} goal={selectedGoal} library="seaborn" />
                    </section>
                )}
            </div>
        )}
      </main>
    </div>
  )
}

export default App
