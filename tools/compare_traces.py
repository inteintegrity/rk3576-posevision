"""Measure detection/pose agreement with a recorded baseline; not ground-truth AP."""
import argparse,pathlib,json,collections,numpy as np
p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--output',required=True);a=p.parse_args()
before=[json.loads(s) for s in pathlib.Path(a.before).read_text().splitlines()];after=[json.loads(s) for s in pathlib.Path(a.after).read_text().splitlines()];assert len(before)==len(after)
def overlap(x,y):
    x,y=np.asarray(x),np.asarray(y);inter=np.maximum(np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2]),0).prod()
    return float(inter/((x[2:]-x[:2]).prod()+(y[2:]-y[:2]).prod()-inter+1e-8))
counts=collections.Counter();matched=collections.Counter();extra=collections.Counter();ious=[];pose_errors=[]
for old,new in zip(before,after):
    assert old['frame']==new['frame'];used=set()
    for d in old['detections']:
        c=d['class_id'];counts[c]+=1
        score,j=max([(overlap(d['bbox'],e['bbox']),j) for j,e in enumerate(new['detections']) if e['class_id']==c and j not in used],default=(0,-1))
        if score>=.5:used.add(j);matched[c]+=1;ious.append(score)
    for j,d in enumerate(new['detections']):
        if j not in used:extra[d['class_id']]+=1
    used=set()
    for t in old['tracks']:
        if not t.get('keypoints'):continue
        score,j=max([(overlap(t['bbox'],n['bbox']),j) for j,n in enumerate(new['tracks']) if n.get('keypoints') and j not in used],default=(0,-1))
        if score<.9:continue
        used.add(j);kp=np.array(t['keypoints']);kq=np.array(new['tracks'][j]['keypoints']);valid=(kp[:,2]>.3)&(kq[:,2]>.3)
        pose_errors.extend(np.linalg.norm(kp[valid,:2]-kq[valid,:2],axis=1).tolist())
result={'frames_compared':len(before),'baseline_detections':dict(counts),'matched_detections':dict(matched),'extra_detections':dict(extra),'retention_against_baseline':{c:matched[c]/n for c,n in counts.items()},'mean_matched_detection_iou':float(np.mean(ious)),'compared_keypoints':len(pose_errors),'mean_keypoint_difference_pixels':float(np.mean(pose_errors)) if pose_errors else None,'note':'Agreement against previous real inference; no manually labeled accuracy claim.'}
pathlib.Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
