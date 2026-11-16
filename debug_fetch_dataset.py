from pathlib import Path
import traceback

try:
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='creditcard', version=1, as_frame=True, parser='auto')
    body = f"SUCCESS\nshape={data.data.shape}\nlabels={data.target.value_counts().to_dict()}\ncolumns={list(data.data.columns[:5])}\n"
except Exception as exc:
    body = 'ERROR\n' + traceback.format_exc()

out = Path('dataset_check.txt')
out.write_text(body, encoding='utf-8')
print('done')
