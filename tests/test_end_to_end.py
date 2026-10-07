"""Real CPU model + local HTTP API; temporary data is deleted after the tests."""
import importlib.util, json, os, socket, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import cv2
ROOT=Path(__file__).resolve().parents[1]

class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        with socket.socket() as s:
            s.bind(('127.0.0.1',0)); cls.port=s.getsockname()[1]
        cls.base=f'http://127.0.0.1:{cls.port}'
        cls.proc=subprocess.Popen([sys.executable,str(ROOT/'server.py'),'--port',str(cls.port)],env={**os.environ,'NEON_DATA':cls.temp.name},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                cls.token=cls.request('/api/config')['token']; break
            except Exception: time.sleep(.1)
        else: raise RuntimeError('Server did not start')
        import ultralytics
        cls.photo=Path(ultralytics.__file__).parent/'assets'/'bus.jpg'
    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate(); cls.proc.wait(timeout=20); cls.temp.cleanup()
    @classmethod
    def request(cls,path,body=None,raw=None,token=True):
        data=raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers={'X-Neon-Token':getattr(cls,'token','')} if token else {}
        with urlopen(Request(cls.base+path,data=data,headers=headers),timeout=60) as res:
            payload=res.read()
            return json.loads(payload) if 'application/json' in res.headers.get('Content-Type','') else payload
    def upload(self,data,name): return self.request('/api/upload?name='+name,raw=data)
    def run_job(self,j,**kwargs):
        options=dict(id=j['id'],confidence=.35,stride=1,size=320);options.update(kwargs)
        self.request('/api/start',options)
        for _ in range(300):
            item=next(i for i in self.request('/api/jobs') if i['id']==j['id'])
            if item['state'] in {'completed','failed','cancelled'}: return item
            time.sleep(.1)
        self.fail('Processing timed out')
    def test_01_image_real_detections_exports(self):
        j=self.run_job(self.upload(self.photo.read_bytes(),'street.jpg'))
        self.assertEqual(j['state'],'completed',j)
        report=self.request(f"/files/{j['id']}/results.json")
        ds=report['samples'][0]['detections']
        self.assertTrue(any(d['label']=='bus' for d in ds),ds)
        self.assertGreater(len(self.request(f"/files/{j['id']}/annotated.jpg")),1000)
        self.assertIn(b'source_frame',self.request(f"/files/{j['id']}/detections.csv"))
        self.request('/api/delete',{'id':j['id']})
    def test_02_video_real_processing(self):
        path=Path(self.temp.name)/'test.avi'; image=cv2.resize(cv2.imread(str(self.photo)),(320,240))
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240))
        for _ in range(6): writer.write(image)
        writer.release()
        j=self.run_job(self.upload(path.read_bytes(),'street.avi'),stride=3)
        self.assertEqual(j['state'],'completed',j)
        report=self.request(f"/files/{j['id']}/results.json")
        self.assertEqual(len(report['samples']),2)
        self.assertEqual([s['frame'] for s in report['samples']],[0,3])
        output=Path(self.temp.name)/'output.avi';output.write_bytes(self.request(f"/files/{j['id']}/annotated.avi"))
        cap=cv2.VideoCapture(str(output));self.assertTrue(cap.read()[0]);self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),10/3,places=2);cap.release()
        self.request('/api/delete',{'id':j['id']})
    def test_03_invalid_media_fails_honestly(self):
        j=self.run_job(self.upload(b'not an image','bad.png'))
        self.assertEqual(j['state'],'failed'); self.assertIn('decoded',j['error'])
        self.request('/api/delete',{'id':j['id']})
    def test_04_guardrails(self):
        with self.assertRaises(HTTPError) as ctx:self.request('/api/upload?name=a.png',raw=b'x',token=False)
        self.assertEqual(ctx.exception.code,403)
        with self.assertRaises(HTTPError) as ctx:self.upload(b'bad','bad.exe')
        self.assertEqual(ctx.exception.code,400)
        with self.assertRaises(HTTPError):self.request('/files/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/../../server.py')
        j=self.upload(self.photo.read_bytes(),'guard.jpg')
        with self.assertRaises(HTTPError):self.request('/api/start',{'id':j['id'],'confidence':2})
        self.request('/api/delete',{'id':j['id']})

if __name__=='__main__':unittest.main(verbosity=2)
