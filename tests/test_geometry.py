import unittest
import numpy as np
from basketball.court import Court,validate_calibration
from basketball.detector import nms
class Geometry(unittest.TestCase):
    def test_projection_uses_feet(self):
        c=Court({'image_points':[[0,0],[1,0],[1,1],[0,1]],'court_points':[[0,0],[1,0],[1,1],[0,1]]})
        ts=[{'class_id':0,'bbox':[10,10,30,80]},{'class_id':1,'bbox':[10,10,30,30]}]
        c.project(ts,(100,100,3));np.testing.assert_allclose(ts[0]['court_position'],[.2,.8]);np.testing.assert_allclose(ts[1]['court_position'],[.2,.2])
    def test_degenerate_calibration_rejected(self):
        with self.assertRaises(ValueError):validate_calibration({'image_points':[[0,0]]*4,'court_points':[[0,0]]*4})
    def test_mapping_disabled_without_evidence(self):
        t=[{'class_id':0,'bbox':[1,2,3,4]}];Court(None).project(t,(100,100,3));self.assertIsNone(t[0]['court_position'])
    def test_nms_keeps_disjoint_boxes(self):
        boxes=np.array([[0,0,10,10],[1,1,11,11],[30,30,40,40]],float)
        self.assertEqual(nms(boxes,np.array([.9,.8,.7]),.5),[0,2])
    def test_lost_camera_registration_withholds_locations(self):
        c=Court({'image_points':[[0,0],[1,0],[1,1],[0,1]],'court_points':[[0,0],[1,0],[1,1],[0,1]]})
        frame=np.zeros((100,100,3),np.uint8)
        c.update_camera(frame,[]);c.update_camera(frame,[])
        self.assertEqual(c.mapping_status,'lost')
        t=[{'class_id':0,'bbox':[1,2,30,40]}];c.project(t,frame.shape);self.assertIsNone(t[0]['court_position'])
if __name__=='__main__':unittest.main()
