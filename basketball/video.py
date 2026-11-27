"""Bounded, ordered video queues. Every decoded frame is consumed exactly once."""
import queue,threading,time,subprocess
import cv2

class VideoReader:
    def __init__(self,path):
        self.cap=cv2.VideoCapture(str(path))
        self.opened=self.cap.isOpened();self.values={key:self.cap.get(key) for key in (cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT)}
        self.queue=queue.Queue(2);self.stop=threading.Event();self.decode_seconds=0.;self.wait_seconds=0.;self.ended=False
        self.thread=threading.Thread(target=self._run,name='video-decode',daemon=True);self.thread.start()
    def isOpened(self):return self.opened
    def get(self,key):return self.values[key]
    def _put(self,item):
        while not self.stop.is_set():
            try:self.queue.put(item,timeout=.1);return
            except queue.Full:pass
    def _run(self):
        try:
            while not self.stop.is_set():
                tick=time.perf_counter();ok,frame=self.cap.read();self.decode_seconds+=time.perf_counter()-tick
                if not ok:break
                self._put(frame)
        except Exception as error:self._put(error)
        finally:self._put(None);self.cap.release()
    def read(self):
        if self.ended:return False,None
        tick=time.perf_counter();item=self.queue.get();self.wait_seconds+=time.perf_counter()-tick
        if isinstance(item,Exception):raise item
        if item is None:self.ended=True;return False,None
        return True,item
    def release(self):
        self.stop.set();self.thread.join()

class VideoEncoder:
    def __init__(self,command,logfile):
        self.log=open(logfile,'w');self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=self.log)
        self.queue=queue.Queue(2);self.stop=threading.Event();self.error=None;self.closed=False;self.write_seconds=0.
        self.thread=threading.Thread(target=self._run,name='video-encode',daemon=True);self.thread.start()
    def _run(self):
        try:
            while not self.stop.is_set():
                frame=self.queue.get()
                if frame is None:break
                tick=time.perf_counter();self.process.stdin.write(frame.tobytes());self.write_seconds+=time.perf_counter()-tick
        except Exception as error:self.error=error
        finally:
            try:self.process.stdin.close()
            except (BrokenPipeError,OSError):pass
    def _put(self,item):
        while True:
            if self.error:raise RuntimeError('MP4 encoding failed') from self.error
            try:self.queue.put(item,timeout=.1);return
            except queue.Full:
                if not self.thread.is_alive():raise RuntimeError('Video encoder stopped')
    def write(self,frame):self._put(frame)
    def close(self):
        self._put(None);self.thread.join();code=self.process.wait();self.closed=True;self.log.close()
        if self.error or code:raise RuntimeError('MP4 encoding failed; inspect encoder.log') from self.error
    def abort(self):
        if self.closed:return
        self.stop.set();self.process.terminate()
        try:self.queue.put_nowait(None)
        except queue.Full:pass
        self.thread.join(timeout=5)
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.closed=True;self.log.close()
