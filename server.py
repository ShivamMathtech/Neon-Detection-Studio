"""Local, uploaded-media object detection. No live sources or tracking."""
from __future__ import annotations
import csv, json, math, mimetypes, os, re, secrets, shutil, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('NEON_DATA', str(ROOT / 'data'))).resolve()
DATA.mkdir(parents=True, exist_ok=True)
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=1)
JOBS = {}
CANCEL = {}
MODEL = None
MAX_UPLOAD = 250 * 1024 * 1024
MAX_SAMPLES = 1500
ACTIVE = {'uploading', 'uploaded', 'queued', 'running'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov', '.avi', '.mkv', '.webm'}
for meta in DATA.glob('*/job.json'):
    try:
        item = json.loads(meta.read_text())
        if item['state'] in {'queued', 'running'}:
            item.update(state='interrupted', error='Server stopped during analysis. Upload again to retry.')
        JOBS[item['id']] = item
    except (ValueError, KeyError, OSError):
        pass

def update(jid, **fields):
    with LOCK:
        JOBS[jid].update(fields)
        folder = DATA / jid
        temp = folder / 'job.tmp'
        temp.write_text(json.dumps(JOBS[jid], allow_nan=False), encoding='utf-8')
        temp.replace(folder / 'job.json')

def detector():
    global MODEL
    if MODEL is None:
        from ultralytics import YOLO
        model_path = ROOT / 'models' / 'yolov8n.pt'
        if not model_path.is_file():
            raise RuntimeError('Model missing: run python scripts/download_model.py')
        MODEL = YOLO(str(model_path))
    return MODEL

def validate_options(obj):
    conf = float(obj.get('confidence', .35))
    stride = int(obj.get('stride', 3))
    size = int(obj.get('size', 640))
    if not math.isfinite(conf) or not .05 <= conf <= .95:
        raise ValueError('Confidence must be between 0.05 and 0.95.')
    if not 1 <= stride <= 30 or size not in {320, 640}:
        raise ValueError('Invalid frame step or inference resolution.')
    return dict(confidence=conf, stride=stride, size=size)

def analyze(jid, options):
    import cv2
    cap = writer = None
    samples, rows = [], []
    folder = DATA / jid
    started = time.perf_counter()
    try:
        if CANCEL[jid].is_set():
            update(jid, state='cancelled')
            return
        update(jid, state='running', message='Loading model', progress=0)
        model = detector()
        source = folder / JOBS[jid]['source']
        is_image = source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
        if is_image:
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError('Image could not be decoded.')
            count, fps = 1, 1.
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise ValueError('Video could not be decoded. Try an H.264 MP4.')
            count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or not .1 <= fps <= 240:
                raise ValueError('Video has invalid frame-rate metadata.')
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video contains no readable frames.')
        h, w = frame.shape[:2]
        if w * h > 24_000_000:
            raise ValueError('Maximum input resolution is 24 megapixels.')
        # All reported coordinates are in this explicitly recorded analysis space.
        scale = min(1., 1280 / max(w, h))
        aw, ah = max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2)
        stride = 1 if is_image else options['stride']
        update(jid, width=aw, height=ah, original_width=w, original_height=h,
               source_fps=fps, total_frames=count, stride=stride, kind='image' if is_image else 'video')
        (folder / 'frames').mkdir(exist_ok=True)
        if not is_image:
            writer = cv2.VideoWriter(str(folder/'annotated.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps/stride, (aw, ah))
            if not writer.isOpened():
                raise RuntimeError('MJPEG video encoder unavailable on this system.')
        idx = 0
        truncated = False
        while frame is not None:
            if CANCEL[jid].is_set():
                break
            if idx % stride == 0:
                if len(samples) >= MAX_SAMPLES:
                    truncated = True
                    break
                frame_small = cv2.resize(frame, (aw, ah))
                tick = time.perf_counter()
                result = model.predict(frame_small, conf=options['confidence'], imgsz=options['size'], device='cpu', verbose=False, max_det=100)[0]
                ms = (time.perf_counter() - tick) * 1000
                detections = []
                for raw in result.boxes.data.cpu().tolist():
                    x1,y1,x2,y2,confidence,cls = raw[:6]
                    box = [max(0,min(aw-1,round(x1))), max(0,min(ah-1,round(y1))), max(0,min(aw-1,round(x2))), max(0,min(ah-1,round(y2)))]
                    label = str(result.names[int(cls)])
                    d = dict(label=label, class_id=int(cls), confidence=round(confidence,5), box=box)
                    detections.append(d)
                    rows.append([idx,round(idx/fps,4),label,round(confidence,5),*box])
                sample_no = len(samples)
                name = f'frames/{sample_no:05d}.jpg'
                if not cv2.imwrite(str(folder/name), frame_small, [cv2.IMWRITE_JPEG_QUALITY,85]):
                    raise RuntimeError('Could not write review image.')
                marked = frame_small.copy()
                for d in detections:
                    x1,y1,x2,y2 = d['box']
                    cv2.rectangle(marked,(x1,y1),(x2,y2),(78,240,40),2)
                    cv2.putText(marked,f"{d['label']} {d['confidence']:.0%}",(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(78,240,40),1,cv2.LINE_AA)
                if writer:
                    writer.write(marked)
                else:
                    cv2.imwrite(str(folder/'annotated.jpg'),marked)
                samples.append(dict(frame=idx,time=round(idx/fps,4),image=name,inference_ms=round(ms,2),detections=detections))
                update(jid, progress=min(99,round((idx+1)/count*100)) if count else 0,
                       processed=len(samples), detection_count=len(rows), message=f'Analyzed {len(samples)} frames')
            if is_image:
                break
            ok, frame = cap.read()
            if not ok:
                frame = None
            idx += 1
        if writer:
            writer.release(); writer=None
        report = dict(job_id=jid, width=aw,height=ah,original_width=w,original_height=h,
                      source_fps=fps,stride=stride,options=options,model='YOLOv8n COCO',
                      elapsed_seconds=round(time.perf_counter()-started,2),truncated=truncated,
                      cancelled=CANCEL[jid].is_set(),samples=samples)
        (folder/'results.json').write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
        with (folder/'detections.csv').open('w',newline='',encoding='utf-8') as f:
            out=csv.writer(f); out.writerow(['source_frame','seconds','class','confidence','x1','y1','x2','y2']); out.writerows(rows)
        state = 'cancelled' if CANCEL[jid].is_set() else 'completed'
        message = 'Stopped at 1,500 sampled frames; results are partial.' if truncated else ('Cancelled; partial results saved.' if state=='cancelled' else 'Analysis complete')
        update(jid,state=state,progress=100 if state=='completed' else JOBS[jid]['progress'],message=message,elapsed_seconds=report['elapsed_seconds'],truncated=truncated)
    except Exception as exc:
        update(jid,state='failed',error=str(exc),message='Analysis failed')
    finally:
        if cap: cap.release()
        if writer: writer.release()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if self.path != '/api/jobs':
            super().log_message(fmt,*args)
    def send_bytes(self, data, content_type, status=200, extra=None):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        for k,v in (extra or {}).items(): self.send_header(k,v)
        self.end_headers()
        self.wfile.write(data)
    def json(self, value, status=200):
        self.send_bytes(json.dumps(value,allow_nan=False).encode(),'application/json',status)
    def trusted(self):
        # Bind to loopback and reject DNS rebinding / cross-origin mutation.
        allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        return self.headers.get('Host') in allowed
    def do_GET(self):
        if not self.trusted(): return self.json({'error':'Invalid host'},403)
        path=unquote(urlparse(self.path).path)
        if path=='/api/config':
            return self.json(dict(token=TOKEN,model_ready=(ROOT/'models/yolov8n.pt').is_file(),max_upload_mb=250,max_samples=MAX_SAMPLES))
        if path=='/api/jobs':
            with LOCK: items=json.loads(json.dumps(list(JOBS.values())))
            return self.json(sorted(items,key=lambda j:j['created'],reverse=True))
        if path.startswith('/files/'):
            parts=path.split('/')
            if len(parts)<4 or not re.fullmatch(r'[a-f0-9]{32}',parts[2]): return self.json({'error':'Not found'},404)
            base=DATA/parts[2]
            file=(base/'/'.join(parts[3:])).resolve()
            if not file.is_relative_to(base) or file.name.startswith('source') or file.name in {'job.json','job.tmp'}:
                return self.json({'error':'Not found'},404)
        else:
            base=ROOT/'web'
            file=(base/('index.html' if path=='/' else path.lstrip('/'))).resolve()
            if not file.is_relative_to(base): return self.json({'error':'Not found'},404)
        if not file.is_file(): return self.json({'error':'Not found'},404)
        ctype=mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
        self.send_response(200); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(file.stat().st_size)); self.send_header('X-Content-Type-Options','nosniff')
        if file.suffix in {'.csv','.avi'}: self.send_header('Content-Disposition',f'attachment; filename="{file.name}"')
        self.end_headers()
        with file.open('rb') as f: shutil.copyfileobj(f,self.wfile)
    def do_POST(self):
        if not self.trusted() or not secrets.compare_digest(self.headers.get('X-Neon-Token',''),TOKEN):
            return self.json({'error':'Refresh the page to renew the session.'},403)
        parsed=urlparse(self.path); path=parsed.path
        try:
            length=int(self.headers.get('Content-Length','0'))
            if path=='/api/upload':
                if not 0<length<=MAX_UPLOAD: return self.json({'error':'File must be between 1 byte and 250 MB.'},413)
                filename=parse_qs(parsed.query).get('name',['upload'])[0]
                ext=Path(filename).suffix.lower()
                if ext not in EXTENSIONS: raise ValueError('Unsupported file type.')
                with LOCK:
                    if sum(j['state'] in ACTIVE for j in JOBS.values())>=3:
                        return self.json({'error':'Maximum 3 pending jobs. Complete or delete one first.'},409)
                    jid=uuid.uuid4().hex; folder=DATA/jid; folder.mkdir()
                    JOBS[jid]=dict(id=jid,name=Path(filename).name[:160],source='source'+ext,created=time.time(),state='uploading',progress=0,processed=0,detection_count=0,message='Uploading')
                try:
                    self.connection.settimeout(120)
                    remaining=length
                    with (folder/('source'+ext)).open('wb') as f:
                        while remaining:
                            block=self.rfile.read(min(1024*1024,remaining))
                            if not block: raise ValueError('Upload was interrupted.')
                            f.write(block); remaining-=len(block)
                    update(jid,state='uploaded',message='Ready to analyze')
                except Exception:
                    with LOCK: JOBS.pop(jid,None)
                    shutil.rmtree(folder,ignore_errors=True)
                    raise
                return self.json(JOBS[jid],201)
            if length>4096: return self.json({'error':'Request too large'},413)
            obj=json.loads(self.rfile.read(length) or b'{}')
            jid=obj.get('id')
            with LOCK:
                if jid not in JOBS: return self.json({'error':'Job not found'},404)
                state=JOBS[jid]['state']
                if path=='/api/start':
                    if state!='uploaded': return self.json({'error':'This upload was already processed.'},409)
                    options=validate_options(obj)
                    CANCEL[jid]=threading.Event(); update(jid,state='queued',options=options,message='Waiting for worker')
                    POOL.submit(analyze,jid,options)
                elif path=='/api/cancel':
                    if jid in CANCEL: CANCEL[jid].set()
                elif path=='/api/delete':
                    if state in {'running','queued','uploading'}: return self.json({'error':'Cancel and wait before deleting.'},409)
                    shutil.rmtree(DATA/jid); JOBS.pop(jid); CANCEL.pop(jid,None)
                    return self.json({'deleted':jid})
                else: return self.json({'error':'Not found'},404)
                return self.json(JOBS[jid])
        except (ValueError,TypeError,KeyError) as exc:
            return self.json({'error':str(exc)},400)
        except Exception as exc:
            return self.json({'error':str(exc)},500)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Neon Detection Studio: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        for event in CANCEL.values(): event.set()
        server.server_close(); POOL.shutdown(wait=True,cancel_futures=True)
