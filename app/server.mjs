import express from 'express';
import { Client } from '@elastic/elasticsearch';
import fetch from 'node-fetch';
import { Ollama } from 'ollama';

// Ollama client
const ollama = new Ollama({ host: 'http://localhost:11434' });
// Elasticsearch client
const es = new Client({ node: 'http://localhost:9200' }); 

// embedding function
async function getEmbedding(query) {
  const response = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'llama3.1:8b',
      prompt: query
    })
  });

  const data = await response.json();
  return data.embedding;
}

// vector search
async function searchVector(embedding) {
    const result = await es.knnSearch({
      index: 'twitter_posts',
      knn: {
        field: 'text_vector',
        k: 10,
        num_candidates: 100,
        query_vector: embedding,
      }
    });
    return result.hits.hits;
}

const app = express(); 
const port = 3000; 

app.use(express.json());


app.use(express.static('public'));


app.post('/chat', async (req, res) => {
    const question = req.body.content;
    const embedding = await getEmbedding(question);
    const topDocs = await searchVector(embedding);

    const context = topDocs.map(doc => doc._source.text).join('\n\n');
    

    const messages = [
        {
          "role": "system",
          "content": `
            you are an assistant and you answer questions asked only in relation to the context
          `,
        },
        {
          "role": "user",
          "content": `
            question: "${question}"
            context \n\n: ${context}.
          `,
        },
      ];

    const response = await ollama.chat({ model: 'llama3.1:8b', messages: messages, stream: true });
    

    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      });
    
      
      for await (const part of response) {
        
        res.write(`data: ${JSON.stringify(part.message)}\n\n`);
      }
    
    res.end();
    
    
});

// On écoute sur le port configuré
app.listen(port, '0.0.0.0', () => {
    console.log(`Server listening : http://127.0.0.1:${port}`);
});
