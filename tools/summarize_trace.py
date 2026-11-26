import argparse,collections,json,pathlib
p=argparse.ArgumentParser();p.add_argument('trace');a=p.parse_args()
counts=collections.Counter();states=collections.Counter();teams=collections.Counter();ids=set();frames=0;ball_frames=0;poses=0;mapped=0;visible_joints=0
for line in pathlib.Path(a.trace).open():
    try:f=json.loads(line)
    except json.JSONDecodeError:break
    frames+=1;states[f.get('mapping_status','unknown')]+=1;seen_ball=False
    for d in f['detections']:
        counts[d['class_id']]+=1;seen_ball|=d['class_id']==1
    ball_frames+=seen_ball
    for t in f['tracks']:
        ids.add(t['id']);mapped+=t.get('court_position') is not None
        if t['class_id']==0:teams[t.get('team',-1)]+=1
        if t.get('keypoints'):
            poses+=1;visible_joints+=sum(k[2]>.3 for k in t['keypoints'])
print(json.dumps({'frames':frames,'detections_by_class':dict(counts),'frames_with_ball_detection':ball_frames,'track_ids':len(ids),'pose_observations':poses,'confidence_qualified_joint_observations':visible_joints,'mapped_track_observations':mapped,'mapping_status_frames':dict(states),'team_observations':dict(teams)},indent=2))
