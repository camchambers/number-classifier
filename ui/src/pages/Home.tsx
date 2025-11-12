import { useState } from 'react';
import { Canvas } from '../components/Canvas';
import { ClassificationResult } from '../components/ClassificationResult';
import { classifyDigit } from '../api/classifier';

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
    <div style={{ maxWidth: 400, margin: '0 auto', padding: 16 }}>
      <h2>Draw a Digit (0-9)</h2>
      <Canvas onImageReady={handleImageReady} />
      {loading && <p>Classifying...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <ClassificationResult digit={digit} confidence={confidence} probabilities={probabilities} />
    </div>
  );
}
