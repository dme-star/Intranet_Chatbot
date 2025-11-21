import os, time, glob
from embedding_store import EmbeddingStore

DATA_DIR = os.environ.get('INTRANET_DATA_DIR', '/data')
os.makedirs(DATA_DIR, exist_ok=True)
store = EmbeddingStore(data_dir=DATA_DIR)
store.load()

def scan_loop():
    print('Ingest worker started, scanning', DATA_DIR)
    known = set()
    while True:
        files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
        for f in files:
            if f not in known:
                print('Ingesting', f)
                try:
                    store.ingest_file(f)
                    known.add(f)
                except Exception as e:
                    print('Failed ingest', f, e)
        time.sleep(60)

if __name__ == '__main__':
    scan_loop()
