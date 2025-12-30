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

import { GoalGenerator } from './components/GoalGenerator';
import { Visualizer } from './components/Visualizer';

// ... interface Summary ...

function App() {
  const [activeModel, setActiveModel] = useState<{provider: string, model: string}>({provider: '', model: ''});
  const [summary, setSummary] = useState<Summary | null>(null);
  const [selectedGoal, setSelectedGoal] = useState<any>(null);
  const [initialGoals, setInitialGoals] = useState<any[]>([]);
  const [initialCharts, setInitialCharts] = useState<any[]>([]);
  const [error, setError] = useState<string>('');

  const handleModelChange = useCallback((provider: string, model: string) => {
    setActiveModel({ provider, model });
    console.log(`Selected: ${provider} - ${model}`);
  }, []);

  const handleUploadSuccess = (data: any) => {
      setSummary(data.summary);
      setInitialGoals(data.goals || []);
      setInitialCharts(data.charts || []);
      setError('');
      setSelectedGoal(null);
  };

  const handleUploadError = (msg: string) => {
      setError(msg);
      setSummary(null);
      setInitialGoals([]);
      setInitialCharts([]);
      setSelectedGoal(null);
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
                <section className="summary-area">
                    <div className="summary-card">
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                            <h3>Dataset Summary</h3>
                            <button className="secondary" onClick={() => { setSummary(null); setSelectedGoal(null); }} style={{fontSize: '0.9rem'}}>Reset / New Upload</button>
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
