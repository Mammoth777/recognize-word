import fs from 'fs';

const csvFilePath = './public/dataset/archive/A_Z Handwritten Data.csv';
const outputFilePath = './public/dataset/archive/shuffled.csv';

async function shuffleCSV() {
  const readStream = fs.createReadStream(csvFilePath, { encoding: 'utf-8' });
  const lines = [];
  
  let buffer = '';
  for await (const chunk of readStream) {
    buffer += chunk;
    const parts = buffer.split('\n');
    buffer = parts.pop();
    lines.push(...parts);
  }
  if (buffer.length > 0) {
    lines.push(buffer);
  }

  // Shuffle lines
  for (let i = lines.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [lines[i], lines[j]] = [lines[j], lines[i]];
  }

  // Write shuffled lines to new file
  const writeStream = fs.createWriteStream(outputFilePath, { encoding: 'utf-8' });
  for (const line of lines) {
    writeStream.write(line + '\n');
  }
  writeStream.end();
}

shuffleCSV().then(() => {
  console.log('CSV file shuffled and saved successfully.');
}).catch(err => {
  console.error('Error shuffling CSV file:', err);
});