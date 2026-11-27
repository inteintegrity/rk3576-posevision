import json, pathlib, time, os
import cv2
import numpy as np
from .detector import Detector
from .tracker import Tracker,Teams
from .pose import Pose,EDGES
from .court import Court
from .runtime import configure_cpu
from .video import VideoReader,VideoEncoder
ROOT=pathlib.Path(__file__).resolve().parents[1]
COLORS={-1:(170,170,170),0:(230,130,50),1:(80,210,100)}

def process(video,output,calibration=None,pose_backend='hybrid',max_frames=0,update=lambda x:None):
    output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();cpu_limits=configure_cpu();cap=VideoReader(video)
    detector=pose=writer=tracker=None;count=0;pose_count=0;track_count=0;class_counts={};mapped=0;mapping_registered=0;mapping_lost=0
    try:
        if not cap.isOpened():raise ValueError('Cannot decode this MP4')
        fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid video frame rate')
        detector_model=pathlib.Path(os.environ.get('BASKETBALL_DETECTOR',ROOT/'models/detector.rknn'))
        detector=Detector(detector_model);tracker=Tracker(ROOT/'models/reid.onnx');teams=Teams();court=Court(calibration)
        if pose_backend!='off':pose=Pose(ROOT/'models'/('pose_backbone.rknn' if pose_backend=='hybrid' else 'pose.rknn' if pose_backend=='rknn' else 'pose.onnx'),pose_backend)
        import imageio_ffmpeg
        stages={'initialization':time.perf_counter()-start}
        def record(name,tick):stages[name]=stages.get(name,0.)+time.perf_counter()-tick
        with (output/'frames.jsonl').open('w') as trace:
            while True:
                tick=time.perf_counter();ok,frame=cap.read();record('decode_wait',tick)
                if not ok:break
                if max_frames and count>=max_frames:break
                # Detection/tracking use original pixels. Render at <=960 px width.
                tick=time.perf_counter();calls_before=len(detector.times);dets=detector.detect(frame);record('detection',tick)
                for d in dets:class_counts[detector.names[int(d.cls[0])]]=class_counts.get(detector.names[int(d.cls[0])],0)+1
                tick=time.perf_counter();tracks=tracker.update(frame,dets);record('tracking_reid',tick)
                tick=time.perf_counter();teams.classify(frame,tracks);record('teams',tick)
                tick=time.perf_counter()
                for t in tracks:
                    if pose and t['class_id']==0:t['keypoints']=pose.estimate(frame,t['bbox']);pose_count+=1
                record('pose',tick);tick=time.perf_counter()
                court.update_camera(frame,dets);court.project(tracks,frame.shape);track_count+=len(tracks)
                mapped+=sum(t['court_position'] is not None for t in tracks)
                mapping_registered+=court.mapping_status in ('calibrated','camera-tracked');mapping_lost+=court.mapping_status=='lost'
                record('camera_projection',tick);tick=time.perf_counter()
                trace.write(json.dumps({'frame':count,'time_s':count/fps,'detection_calls':len(detector.times)-calls_before,'mapping_status':court.mapping_status,'mapping_inliers':court.inliers,'homography':court.H.tolist() if court.H is not None and court.mapping_status!='lost' else None,'detections':[{'bbox':d.xyxy[0].tolist(),'class_id':int(d.cls[0]),'confidence':float(d.conf[0])} for d in dets],'tracks':tracks})+'\n')
                record('trace_json',tick);tick=time.perf_counter()
                h,w=frame.shape[:2];vw=min(w,960);vh=round(h*vw/w)//2*2
                canvas=cv2.resize(frame,(vw,vh));sx,sy=vw/w,vh/h
                def xy(p):return (int(p[0]*sx),int(p[1]*sy))
                # Draw all detections immediately, even before track confirmation.
                for d in dets:
                    box=d.xyxy[0];cv2.rectangle(canvas,xy(box[:2]),xy(box[2:]),(140,140,140),1)
                for t in tracks:
                    color=(30,180,255) if t['class_id']==1 else COLORS[t.get('team',-1)]
                    x1,y1=xy(t['bbox'][:2]);x2,y2=xy(t['bbox'][2:]);cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    label=f"{['P','B','R'][t['class_id']]}{t['id'].split('-')[1]}"
                    if t['class_id']==0:label+=f" T{t.get('team',-1)+1}" if t.get('team',-1)>=0 else ' T?'
                    cv2.putText(canvas,label,(max(x1,0),max(y1-8,20)),0,.5,color,1)
                    points=t.get('keypoints',[])
                    for a,b in EDGES:
                        if points and min(points[a][2],points[b][2])>.3:cv2.line(canvas,xy(points[a][:2]),xy(points[b][:2]),(255,255,80),2)
                    for x,y,score in points:
                        if score>.3:cv2.circle(canvas,xy((x,y)),2,(60,80,255),-1)
                side=cv2.resize(court.draw(tracks),(400,vh));canvas=np.concatenate([canvas,side],axis=1)
                record('render',tick);tick=time.perf_counter()
                if writer is None:
                    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s',f'{vw+400}x{vh}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','ultrafast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'result.mp4')]
                    cmd[cmd.index('-preset'):cmd.index('-preset')]=['-threads','2']
                    writer=VideoEncoder(cmd,output/'encoder.log')
                writer.write(canvas);count+=1;record('encoder_submit',tick)
                if count%5==0 or count==1:
                    tick=time.perf_counter();cv2.imwrite(str(output/'preview.tmp.jpg'),canvas);(output/'preview.tmp.jpg').replace(output/'preview.jpg')
                    record('preview',tick)
                    update({'frames':count,'total':min(total,max_frames) if max_frames else total,'detection_npu_fps':len(detector.times)/sum(detector.times),'detection_npu_frame_fps':count/sum(detector.times),'detection_npu_calls':len(detector.times),'pose_npu_fps':pose.calls/pose.seconds if pose and pose.backend in ('rknn','hybrid') and pose.calls else None,'reid_npu_fps':tracker.npu_calls/tracker.npu_seconds if tracker.npu_calls else None,'end_to_end_fps':count/(time.perf_counter()-start)})
        if not count:raise ValueError('Video contains no decodable frames')
        tick=time.perf_counter();writer.close();record('encoder_finalize',tick)
        elapsed=time.perf_counter()-start
        npu_seconds=sum(detector.times)+(pose.seconds if pose and pose.backend in ('rknn','hybrid') else 0)+tracker.npu_seconds
        report={'frames':count,'source_fps':fps,'source_total_frames':total,'partial_run':bool(max_frames and count<total),'elapsed_seconds':elapsed,'end_to_end_fps':count/elapsed,'detection_npu_fps':len(detector.times)/sum(detector.times),'detection_npu_frame_fps':count/sum(detector.times),'detection_npu_calls':len(detector.times),'detection_npu_seconds':sum(detector.times),'pose_backend':pose_backend,'pose_calls':pose_count,'pose_cpu_head_seconds':pose.head_seconds if pose else 0,'pose_npu_fps':pose.calls/pose.seconds if pose and pose.backend in ('rknn','hybrid') and pose.calls else None,'npu_pipeline_fps':count/npu_seconds,'npu_seconds':npu_seconds,'track_observations':track_count,'calibration':calibration,'mapping':'manual homography with floor optical-flow/RANSAC camera compensation' if calibration else 'disabled: calibration required','mapping_status':court.mapping_status,'mapping_registered_frames':mapping_registered,'mapping_lost_frames':mapping_lost,'mapped_observations':mapped,'detections_by_class':class_counts,'ball_tiling':'adaptive human-centered vertical band, square tiles up to 1440px; same weights; ball confidence 0.12','reid_backend':tracker.backend,'timing_scope':'NPU FPS measures blocking RKNN inference API only. End-to-end includes initialization, decode, preprocessing, CPU association/ReID, pose, projection, drawing, JSON and MP4 finalization; excludes upload.'}
        report.update(reid_backend=tracker.backend,reid_npu_seconds=tracker.npu_seconds,reid_npu_calls=tracker.npu_calls,reid_npu_fps=tracker.npu_calls/tracker.npu_seconds if tracker.npu_calls else None,reid_total_seconds=tracker.reid_seconds,association_seconds=tracker.association_seconds,opencv_threads=cv2.getNumThreads(),blas_threads=1,deepsort_feature_budget=100,decode_worker_seconds=cap.decode_seconds,decode_wait_seconds=cap.wait_seconds,encoder_worker_write_seconds=writer.write_seconds,video_queue_frames=2,stage_wall_seconds=stages,detector_model=detector_model.name,detector_core_mask=detector.core_mask,reid_core_mask=getattr(tracker,'core_mask',None))
        (output/'report.json').write_text(json.dumps(report,indent=2));update(report);return report
    finally:
        cap.release()
        if writer:writer.abort()
        if detector:detector.close()
        if pose:pose.close()
        if tracker:tracker.close()
        cpu_limits.restore_original_limits()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--video',default=str(ROOT/'upstream/input/sample2.mp4'));p.add_argument('--output',required=True);p.add_argument('--pose',choices=['hybrid','rknn','onnx','off'],default='hybrid');p.add_argument('--max-frames',type=int,default=0);p.add_argument('--calibration');a=p.parse_args()
    print(json.dumps(process(a.video,a.output,json.loads(pathlib.Path(a.calibration).read_text()) if a.calibration else None,a.pose,a.max_frames,lambda s:print(json.dumps(s),flush=True)),indent=2))
