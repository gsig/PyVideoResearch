from metrics.metric import Metric
import numpy as np
import os
import sys
import pprint


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def tensor2str(tensor):
    return ''.join([chr(y) for y in tensor])


class FRCNNMetric(Metric):
    def __init__(self):
        this_dir = os.path.dirname(__file__)
        lib_path = os.path.join(this_dir, '../external/ActivityNet/Evaluation')
        sys.path.insert(0, lib_path)
        from ava import object_detection_evaluation
        from ava import standard_fields
        sys.path.pop(0)
        self.sf = standard_fields
        categories = [{"id": i, "name": i} for i in range(1, 81)]
        self.evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

    def update(self, predictions, targets):
        try:
            for t in targets:
                image_key = make_image_key(t['vid'], t['start'])
                self.evaluator.add_single_ground_truth_image_info(
                    image_key, {
                        self.sf.InputDataFields.groundtruth_boxes:
                            np.array(t['boxes'], dtype=float),
                        self.sf.InputDataFields.groundtruth_classes:
                            np.array(t['labels'] + 1, dtype=int),
                        self.sf.InputDataFields.groundtruth_difficult:
                            np.zeros(len(t['labels']), dtype=bool)
                    })
            for p in predictions:
                image_key = make_image_key(tensor2str(p['vid']), p['start'])
                self.evaluator.add_single_detected_image_info(
                    image_key, {
                        self.sf.DetectionResultFields.detection_boxes:
                            np.array(p['boxes'], dtype=float),
                        self.sf.DetectionResultFields.detection_classes:
                            np.array(p['labels'] + 1, dtype=int),
                        self.sf.DetectionResultFields.detection_scores:
                            np.array(p['scores'], dtype=float)
                    })
        except Exception as e:
            print(e)

    def compute(self):
        metrics = self.evaluator.evaluate()
        pprint.pprint(metrics, indent=2)
        return ('AVA', metrics['PascalBoxes_Precision/mAP@0.5IOU'])
