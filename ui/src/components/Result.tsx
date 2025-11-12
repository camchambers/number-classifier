import React from 'react';

interface ResultProps {
  digit: number | null;
  confidence: number | null;
  probabilities: Record<number, number> | null;
}

const Result: React.FC<ResultProps> = ({ digit, confidence, probabilities }) => {
  if (digit === null || confidence === null || !probabilities) return null;

  return (
    <div style={{ marginTop: 16 }}>
      <h3>Prediction</h3>
      <p>Digit: <strong>{digit}</strong></p>
      <p>Confidence: <strong>{(confidence * 100).toFixed(2)}%</strong></p>
      <h4>Probabilities:</h4>
      <ul>
        {Object.entries(probabilities).map(([d, prob]) => (
          <li key={d}>
            {d}: {(prob * 100).toFixed(2)}%
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Result;
