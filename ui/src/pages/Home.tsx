import React, { useState } from 'react';
import Canvas from '../components/Canvas';
import Result from '../components/Result';
import { classifyDigit } from '../api/classifier';

const Home: React.FC = () => {
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
    } catch (e) {
      setError('Failed to classify digit.');
      // Optionally log error: console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: '0 auto', padding: 16 }}>
      <h2>Draw a Digit (0-9)</h2>
      <Canvas onImageReady={handleImageReady} />
      {loading && <p>Classifying...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <Result digit={digit} confidence={confidence} probabilities={probabilities} />
    </div>
  );
};

export default Home;
