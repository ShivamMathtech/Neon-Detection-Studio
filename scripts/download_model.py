from pathlib import Path
from urllib.request import urlretrieve
import hashlib
root = Path(__file__).resolve().parents[1]
path = root / 'models' / 'yolov8n.pt'
path.parent.mkdir(exist_ok=True)
if path.exists():
    print('Model already present:', path)
else:
    temp = path.with_suffix('.part')
    urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt', temp)
    if temp.stat().st_size < 1_000_000:
        temp.unlink(); raise RuntimeError('Download is incomplete')
    temp.replace(path)
    print('Downloaded:', path)
print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
