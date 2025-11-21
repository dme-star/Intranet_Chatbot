import os, json
from sentence_transformers import SentenceTransformer
import numpy as np, hashlib

MODEL_NAME = 'all-MiniLM-L6-v2'
def sha256(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

class EmbeddingStore:
    def __init__(self, data_dir='/data'):
        self.data_dir = data_dir
        self.vectors_path = os.path.join(self.data_dir, 'vectors.json')
        self.index = []
        self.model = None

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME)
        if os.path.exists(self.vectors_path):
            try:
                with open(self.vectors_path, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                for item in self.index:
                    item['embedding'] = np.array(item['embedding'], dtype=float)
            except Exception as e:
                print('failed load', e)
                self.index = []

    def save(self):
        serial = []
        for item in self.index:
            serial.append({'id': item['id'], 'source': item['source'], 'text': item['text'], 'embedding': item['embedding'].tolist()})
        with open(self.vectors_path, 'w', encoding='utf-8') as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)

    def chunk_text(self, text, max_chars=1000, overlap=200):
        chunks=[]
        i=0
        while i < len(text):
            chunks.append(text[i:i+max_chars])
            i += max_chars - overlap
        return chunks

    def ingest_file(self,path):
        with open(path,'r',encoding='utf-8') as f:
            text = f.read()
        chunks = self.chunk_text(text)
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME)
        for idx,ch in enumerate(chunks):
            cid = sha256(path + '::' + str(idx))
            emb = self.model.encode(ch).tolist()
            found=False
            for it in self.index:
                if it['id']==cid:
                    it['text']=ch; it['embedding']=np.array(emb); it['source']=os.path.basename(path); found=True; break
            if not found:
                self.index.append({'id':cid,'source':os.path.basename(path),'text':ch,'embedding':np.array(emb)})
        self.save()
