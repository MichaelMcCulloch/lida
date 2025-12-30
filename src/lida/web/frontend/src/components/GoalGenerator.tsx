import React, { useState } from 'react';

interface Goal {
    question: string;
    visualization: string;
    rationale: string;
    index: number;
}

interface Summary {
    name: string;
    file_name: string;
    dataset_description: string;
    field_names: string[];
}

interface GoalGeneratorProps {
    summary: Summary;
    onGoalSelect: (goal: Goal) => void;
    initialGoals?: Goal[];
}

export const GoalGenerator: React.FC<GoalGeneratorProps> = ({ summary, onGoalSelect, initialGoals = [] }) => {
    const [goals, setGoals] = useState<Goal[]>(initialGoals);
    const [loading, setLoading] = useState(false);
    const [n, setN] = useState(2);

    React.useEffect(() => {
        if (initialGoals && initialGoals.length > 0) {
            setGoals(initialGoals);
        }
    }, [initialGoals]);

    const generateGoals = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/goal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    summary: summary,
                    n: n,
                    textgen_config: { n: 1, temperature: 0 }
                })
            });
            const data = await res.json();
            if (data.status) {
                setGoals(data.data);
            } else {
                console.error(data.message);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="goal-generator">
            <div className="goal-controls">
                <label>
                    Number of Goals:
                    <input 
                        type="number" 
                        min="1" 
                        max="10" 
                        value={n} 
                        onChange={(e) => setN(parseInt(e.target.value))} 
                    />
                </label>
                <button onClick={generateGoals} disabled={loading}>
                    {loading ? 'Generating...' : 'Generate Goals'}
                </button>
            </div>

            <div className="goal-list">
                {goals.map((goal, idx) => (
                    <div key={idx} className="goal-card" onClick={() => onGoalSelect(goal)}>
                        <h4>{goal.question}</h4>
                        <p>{goal.rationale}</p>
                        <div className="viz-code-preview">
                            <small>{goal.visualization.slice(0, 50)}...</small>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
