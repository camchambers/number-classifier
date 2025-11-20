interface ResultProps {
  digit: number | null;
  confidence: number | null;
  probabilities: Record<number, number> | null;
}

export const ClassificationResult = ({ digit, confidence, probabilities }: ResultProps) => {
  if (digit === null || confidence === null || !probabilities) return null;

  return (
    <div className="result-container">
      <div className="prediction">
        <div className="prediction-digit">{digit}</div>
        <div className="prediction-confidence">{(confidence * 100).toFixed(1)}% confidence</div>
      </div>
      <div className="probabilities">
        <h4>All Probabilities</h4>
        <div className="probability-bars">
          {Object.entries(probabilities)
            .sort(([a], [b]) => Number(a) - Number(b))
            .map(([d, prob]) => (
              <div key={d} className="probability-item">
                <span className="probability-label">{d}</span>
                <div className="probability-bar-container">
                  <div 
                    className="probability-bar" 
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
                <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};