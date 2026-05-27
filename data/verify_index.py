import sqlite3
c = sqlite3.connect('data/index/rag.db')
print('total docs:', c.execute('SELECT COUNT(*) FROM documents').fetchone()[0])
print('total chunks:', c.execute('SELECT COUNT(*) FROM chunks').fetchone()[0])
print('total chunks with embeddings:', c.execute('SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL').fetchone()[0])
print()
print('demo docs:')
for r in c.execute("SELECT doc_id, json_extract(metadata_json,'$.skill') AS skill, title FROM documents WHERE json_extract(metadata_json,'$.synthetic')=1 ORDER BY skill, doc_id").fetchall():
    print(f"  [{r[1]:7}] {r[0]:30} {r[2]}")
