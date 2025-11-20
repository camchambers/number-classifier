import { useRef, useState, useEffect } from 'react';

interface CanvasProps {
  onImageReady: (img: string) => void;
}

export const Canvas = ({ onImageReady }: CanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [drawing, setDrawing] = useState(false);

  // Initialize canvas with white background and black brush
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Set white background
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Set black brush - thicker for better recognition
        ctx.strokeStyle = '#000000';
        ctx.fillStyle = '#000000';
        ctx.lineWidth = 15;  // Thicker line for better recognition
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
      }
    }
  }, []);

  const startDrawing = (e: React.MouseEvent) => {
    setDrawing(true);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx?.beginPath();
      ctx?.moveTo(x, y);
    }
  };

  const draw = (e: React.MouseEvent) => {
    if (!drawing) return;
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx?.lineTo(x, y);
      ctx?.stroke();
    }
  };

  const stopDrawing = () => {
    setDrawing(false);
  };

  const handleClear = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      // Clear with white background
      ctx!.fillStyle = '#FFFFFF';
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      ctx?.fillRect(0, 0, canvas.width, canvas.height);
    }
  };

  const handleSubmit = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      // Don't invert - training data is black on white, same as our canvas
      const img = canvas.toDataURL('image/png');
      onImageReady(img);
    }
  };

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        className="canvas"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      />
      <div className="button-group">
        <button className="btn btn-secondary" onClick={handleClear}>Clear</button>
        <button className="btn btn-primary" onClick={handleSubmit}>Classify</button>
      </div>
    </div>
  );
};

export default Canvas;
