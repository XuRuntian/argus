"""Half-open, single-track segment metrics with one-to-one temporal matching."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    overlap = max(0, min(a['end_frame'], b['end_frame']) - max(a['start_frame'], b['start_frame']))
    return overlap / (max(a['end_frame'], b['end_frame']) - min(a['start_frame'], b['start_frame']))


def matches(truth, prediction, threshold):
    if not truth or not prediction:
        return []
    # Cardinality dominates IoU: maximize valid matches first, overlap second.
    weights = np.zeros((len(truth), len(prediction)))
    for i, a in enumerate(truth):
        for j, b in enumerate(prediction):
            overlap = iou(a, b)
            if a['action_id'] == b['action_id'] and a['action_id'] is not None and overlap >= threshold:
                weights[i, j] = min(len(truth), len(prediction)) + 1 + overlap
    left, right = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(j)) for i, j in zip(left, right) if weights[i, j] > 0]


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def score_episode(truth, prediction, fps):
    if not truth:
        raise ValueError('cannot score an episode without annotated coverage')
    covered = sum(x['end_frame'] - x['start_frame'] for x in truth)
    correct = sum(max(0, min(a['end_frame'], b['end_frame']) - max(a['start_frame'], b['start_frame']))
                  for a in truth for b in prediction if a['action_id'] == b['action_id'])
    result = {'annotated_frames': covered, 'correct_frames': correct,
              'frame_accuracy': correct / covered, 'truth_segments': len(truth),
              'prediction_segments': len(prediction),
              'unknown_segments': sum(p['action_id'] is None for p in prediction)}
    for threshold in (0.5, 0.75):
        paired = matches(truth, prediction, threshold)
        tp = len(paired)
        result[f'segment_{threshold}'] = {'tp': tp, 'fp': len(prediction) - tp,
                                        'fn': len(truth) - tp,
                                        'f1': 2 * tp / (len(truth) + len(prediction))}
        if threshold == 0.5:
            result['boundary_matches'] = tp
            for edge in ('start', 'end'):
                errors = [abs(truth[i][edge + '_frame'] - prediction[j][edge + '_frame']) / fps
                          for i, j in paired]
                result[edge + '_mae_s'] = sum(errors) / tp if tp else None
    result['sequence_edit_distance'] = edit_distance([x['action_id'] for x in truth],
                                                     [x['action_id'] for x in prediction])
    return result


def summarize(rows):
    if not rows:
        raise ValueError('empty evaluation')
    return {'episodes': len(rows),
            'macro_frame_accuracy': sum(x['frame_accuracy'] for x in rows) / len(rows),
            'micro_frame_accuracy': sum(x['correct_frames'] for x in rows) / sum(x['annotated_frames'] for x in rows),
            'macro_segment_f1_0.5': sum(x['segment_0.5']['f1'] for x in rows) / len(rows),
            'macro_segment_f1_0.75': sum(x['segment_0.75']['f1'] for x in rows) / len(rows),
            'missing_predictions': sum(x.get('missing_prediction', False) for x in rows),
            'sequence_edit_distance_total': sum(x['sequence_edit_distance'] for x in rows)}
