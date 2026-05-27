import pyarrow.parquet as pq
import json
import pandas as pd

qt = pq.read_table(r'data\raw\questions_test.parquet').to_pandas()
print('total rows:', len(qt))
print('columns:', list(qt.columns))

def parse_expected(s):
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    text = str(s).strip().replace("'", '"')
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []

has_gold = qt['expected_doc_ids'].apply(lambda s: bool(parse_expected(s)))
print('rows with gold:', int(has_gold.sum()))
print('rows without gold:', int((~has_gold).sum()))
print()
print('distribution of has_gold by index ranges:')
for start in range(0, 500, 50):
    end = start + 50
    sl = has_gold.iloc[start:end]
    print(f'  rows [{start:3d}:{end:3d}]: {int(sl.sum()):2d} with gold, {int((~sl).sum()):2d} without')

print()
print('question_type counts:')
print(qt['question_type'].value_counts().to_string())

print()
print('first 5 rows with empty expected_doc_ids (sample):')
empty_rows = qt[~has_gold].head(5)
for _, r in empty_rows.iterrows():
    print(f'  qid={r.question_id}  type={r.question_type}  expected={r.expected_doc_ids!r}  q={r.question[:80]}')
