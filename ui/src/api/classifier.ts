// API client for digit classification

export async function classifyDigit(base64Image: string) {
  const response = await fetch('http://localhost:8000/predict/base64', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: base64Image }),
  });
  if (!response.ok) throw new Error('API error');
  return response.json();
}
