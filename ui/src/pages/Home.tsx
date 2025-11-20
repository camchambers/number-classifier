import { useState } from 'react';
import { Canvas } from '@components/Canvas';
import { classifyDigit } from '@api/classifier';

export const Home = () => {
  const [digit, setDigit] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<Record<number, number> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageReady = async (img: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await classifyDigit(img);
      setDigit(result.predicted_digit);
      setConfidence(result.confidence);
      setProbabilities(result.probabilities);
    } catch {
      setError('Failed to classify digit.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Number Classifier</h1>
      <p className="subtitle">Draw a digit from 0-9</p>
      
      <div className="main-content">
        <div className="left-panel">
          <Canvas onImageReady={handleImageReady} />
          {loading && <p className="loading">Classifying...</p>}
          {error && <p className="error">{error}</p>}
          {(digit !== null && confidence !== null) && (
            <div className="prediction">
              <div className="prediction-digit">{digit}</div>
              <div className="prediction-confidence">{(confidence * 100).toFixed(1)}% confidence</div>
            </div>
          )}
        </div>
        
        {probabilities && (
          <div className="right-panel">
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
        )}
      </div>
    </div>
  );
}
