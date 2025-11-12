/**
 * Convert canvas data to properly formatted image for the model
 * The model expects white text on black background (MNIST format)
 * But the canvas draws black on white, so we need to invert
 */
export function canvasToInvertedBase64(canvas: HTMLCanvasElement): string {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  
  const tempCtx = tempCanvas.getContext('2d');
  if (!tempCtx) throw new Error('Failed to get canvas context');
  
  // Draw white background
  tempCtx.fillStyle = '#FFFFFF';
  tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
  
  // Draw inverted image (black becomes white)
  const imageData = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  // Invert colors: if pixel is dark (near black), make it white in the output
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    
    // Calculate grayscale intensity
    const intensity = (r + g + b) / 3;
    
    // If pixel is dark (drawn on), make it white
    if (intensity < 128) {
      data[i] = 255;     // R
      data[i + 1] = 255; // G
      data[i + 2] = 255; // B
    } else {
      data[i] = 0;       // R
      data[i + 1] = 0;   // G
      data[i + 2] = 0;   // B
    }
  }
  
  tempCtx.putImageData(imageData, 0, 0);
  return tempCanvas.toDataURL('image/png');
}
