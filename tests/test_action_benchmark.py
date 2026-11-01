import copy
import json

import pytest

from action_benchmark.__main__ import evaluate
from action_benchmark.evaluate import matches, score_episode
from label.action_contract import action_id, catalog, digest, prompt, validate_segments
from label.atomic import write_atomic
from label.harness import read_reply


@pytest.fixture
def contract():
    action = {'subject': 'left_effector', 'skill': 'pick', 'slots': {'object': 'cup'}, 'text': 'pick cup'}
    schema = {'annotation_spec_version': 'skill_spec_v1', 'skills': [{'actions': [action]}],
              'episode_annotation': [{'SECRET_ANSWER': 'frames 17 to 29'}], 'task_description': 'SECRET_ORDER'}
    vocabulary = {'version': 'skill_spec_v1', 'skills': {'pick': {'completion_rule': 'lift completed'}}}
    return {**catalog(schema, vocabulary), 'frame_count': 100, 'fps': 10}


def segment(contract, start=0, end=100):
    return {'start_frame': start, 'end_frame': end, 'action_id': contract['actions'][0]['action_id']}


def test_published_benchmark_preserves_unknown_reason(tmp_path, contract):
    from label.pieces import _publish_reply

    rows = [segment(contract, end=50),
            {'start_frame': 50, 'end_frame': 100, 'action_id': None,
             'reason': 'The supplied clip ends.'}]
    validate_segments(rows, contract)
    record = {'labels': {'constrained_subtasks': rows,
                         'description': 'The supplied clip ends.'}}
    source, target = tmp_path / 'raw.json', tmp_path / 'published.json'
    write_atomic(source, record)
    _publish_reply(source, target)

    published = json.loads(target.read_text())
    assert published['labels']['constrained_subtasks'] == rows
    validate_segments(published['labels']['constrained_subtasks'], contract)
    assert published['labels']['description'] is None
    assert json.loads(source.read_text()) == record


def test_dashscope_pilot_caps_dispatch_and_retains_timeout(tmp_path, monkeypatch):
    from action_benchmark.dashscope_pilot import PilotClient
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-secret')
    client = PilotClient(tmp_path / 'limited', max_calls=1, cap_cny=0.001)
    with pytest.raises(ValueError, match='limit reached'):
        client.send([{'role': 'user', 'content': 'hello'}])
    assert not list(client.out.glob('call_*.json'))

    class BrokenClient:
        def __init__(self, **kwargs):
            assert kwargs['follow_redirects'] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            raise TimeoutError('unknown response')

    monkeypatch.setattr('action_benchmark.dashscope_pilot.httpx.Client', BrokenClient)
    client = PilotClient(tmp_path / 'timeout', max_calls=1, cap_cny=1)
    with pytest.raises(TimeoutError):
        client.send([{'role': 'user', 'content': 'hello'}])
    receipt = (client.out / 'call_001.json').read_text()
    assert 'outcome_unknown' in receipt and 'test-secret' not in receipt
    with pytest.raises(ValueError, match='stopped after'):
        client.send([{'role': 'user', 'content': 'hello'}])

    class UnauthorizedClient(BrokenClient):
        def post(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(status_code=401, text='invalid test-secret')

    monkeypatch.setattr('action_benchmark.dashscope_pilot.httpx.Client', UnauthorizedClient)
    client = PilotClient(tmp_path / 'unauthorized', max_calls=2, cap_cny=1)
    with pytest.raises(ValueError, match='HTTP 401'):
        client.send([{'role': 'user', 'content': 'hello'}])
    with pytest.raises(ValueError, match='stopped after'):
        client.send([{'role': 'user', 'content': 'hello'}])
    assert client.calls == 1
    assert 'test-secret' not in (client.out / 'call_001.json').read_text()


def test_catalog_contains_definitions_not_answers(contract):
    text = prompt(contract)
    assert 'SECRET' not in text
    assert 'lift completed' in text
    assert 'pick cup' in text
    assert contract['actions'][0]['action_id'] == action_id({'subject': 'left_effector', 'skill': 'pick',
                                                           'slots': {'object': 'cup'}, 'text': 'ignored'})
    assert contract['actions'][0]['action_id'] != action_id({'subject': 'right_effector', 'skill': 'pick',
                                                           'slots': {'object': 'cup'}})


@pytest.mark.parametrize('skill', ['hand_over', 'hold', 'return_to_initial_pose'])
def test_missing_rules_use_versioned_defaults_without_overriding_vocabulary(skill):
    schema = {'annotation_spec_version': 'skill_spec_v1', 'skills': [{'actions': [
        {'subject': 'left_effector', 'skill': skill, 'slots': {}, 'text': skill}]}]}
    vocabulary = {'version': 'skill_spec_v1', 'skills': {}}
    original = copy.deepcopy(vocabulary)
    defaults = catalog(schema, vocabulary)
    row = defaults['actions'][0]
    assert vocabulary == original
    assert row['rule_source'] == 'benchmark_defaults_v1'
    assert row['start_rule'] and row['completion_rule']
    assert row['completion_rule'] in prompt({**defaults, 'frame_count': 100, 'fps': 10})
    vocabulary['skills'][skill] = {'completion_rule': 'Authoritative source boundary'}
    upstream = catalog(schema, vocabulary)
    assert upstream['actions'][0]['rule_source'] == 'vocabulary'
    assert upstream['actions'][0]['completion_rule'] == 'Authoritative source boundary'
    assert upstream['actions'][0]['action_id'] == row['action_id']
    assert upstream['catalog_sha256'] != defaults['catalog_sha256']


@pytest.mark.parametrize('changes', [{'action_id': 'invented'}, {'start_frame': -1}, {'end_frame': 101},
                                    {'start_frame': True}, {'end_frame': 0}, {'start_frame': 1.2}])
def test_invalid_output_is_rejected_not_repaired(contract, changes):
    row = {**segment(contract), **changes}
    with pytest.raises(ValueError):
        validate_segments([row], contract)
    labels, ok = read_reply(json.dumps({'task_summary': 'picked cup', 'constrained_subtasks': [row]}),
                            {'action_contract': contract})
    assert not ok
    assert labels['constrained_subtasks'] == [row]
    assert 'closed action contract' in labels['_incomplete'][0]


def test_unknown_and_repeated_segments(contract):
    rows = [segment(contract, 0, 25), {'start_frame': 25, 'end_frame': 50, 'action_id': None, 'reason': 'occluded'},
            segment(contract, 50, 100)]
    assert validate_segments(rows, contract) == rows
    with pytest.raises(ValueError, match='explicit action_id'):
        validate_segments([{'start_frame': 0, 'end_frame': 100, 'reason': 'unknown'}], contract)
    with pytest.raises(ValueError, match='overlapping'):
        validate_segments([segment(contract), segment(contract)], contract)
    altered = copy.deepcopy(contract)
    altered['actions'][0]['primary_action']['subject'] = 'right_effector'
    with pytest.raises(ValueError, match='digest'):
        validate_segments([], altered)


def test_perfect_missing_wrong_and_fragmented_predictions(contract):
    truth = [segment(contract)]
    assert score_episode(truth, truth, 10)['segment_0.75']['f1'] == 1
    missing = score_episode(truth, [], 10)
    assert missing['segment_0.5'] == {'tp': 0, 'fp': 0, 'fn': 1, 'f1': 0}
    assert missing['start_mae_s'] is None
    fragmented = [segment(contract, 0, 50), segment(contract, 50, 100)]
    scores = score_episode(truth, fragmented, 10)
    assert scores['frame_accuracy'] == 1
    assert scores['segment_0.5']['f1'] == pytest.approx(2 / 3)
    assert len(matches(truth, fragmented, .5)) == 1
    assert scores['segment_0.75']['f1'] == 0
    wrong = [{**segment(contract), 'action_id': 'other'}]
    assert score_episode(truth, wrong, 10)['frame_accuracy'] == 0
    assert score_episode(truth, wrong, 10)['sequence_edit_distance'] == 1


def test_boundary_errors_and_gap_mask(contract):
    truth = [segment(contract, 10, 90)]
    scores = score_episode(truth, [segment(contract, 20, 100)], 10)
    assert scores['frame_accuracy'] == 70 / 80
    assert scores['start_mae_s'] == 1
    assert scores['end_mae_s'] == 1


def test_evaluation_counts_missing_episodes_and_rejects_wrong_provenance(tmp_path, contract):
    data = tmp_path / 'source'
    source = {'episode_annotation': [{'id': 0, 'frame_count': 100, 'subtasks': [
        {'start_frame': 0, 'end_frame': 100, 'primary_action': contract['actions'][0]['primary_action']}]}]}
    write_atomic(data / 'lite_annotations/annotation_schema_v1.json', source)
    base = {k: v for k, v in contract.items() if k not in ('frame_count', 'fps')}
    write_atomic(tmp_path / 'manifest.json', {'dataset_id': 'd', 'dataset_path': str(data),
                 'catalog': base, 'reference_sha256': digest(source), 'input_modality': 'multi_view_rgb_only',
                 'episodes': [{'episode_id': 0, 'frame_count': 100, 'fps': 10}]})
    predictions = {'version': 1, 'dataset_id': 'd', 'catalog_sha256': base['catalog_sha256'], 'episodes': []}
    report = evaluate(tmp_path, predictions)
    assert report['summary']['missing_predictions'] == 1
    assert report['summary']['macro_segment_f1_0.5'] == 0
    predictions['episodes'] = [{'episode_id': 0, 'frame_count': 100, 'fps': 10, 'subtasks': [segment(contract)]}]
    assert evaluate(tmp_path, predictions)['summary']['macro_segment_f1_0.5'] == 1
    predictions['episodes'].append(predictions['episodes'][0])
    with pytest.raises(ValueError, match='duplicate'):
        evaluate(tmp_path, predictions)
    predictions['dataset_id'] = 'another'
    with pytest.raises(ValueError, match='mismatch'):
        evaluate(tmp_path, predictions)


def test_agent_replay_binds_exact_prompt_once(tmp_path):
    from action_benchmark.agent_pilot import ReplyReplay

    write_atomic(tmp_path / 'requests/call_001/messages.json', [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'original prompt'}]}])
    write_atomic(tmp_path / 'replies/call_001.json', {'state': 'normal'})
    client = ReplyReplay(tmp_path)
    with pytest.raises(ValueError, match='unknown or repeated'):
        client.annotate_segment({'prompt': 'modified prompt'})
    assert client.annotate_segment({'prompt': 'original prompt'}) == {'state': 'normal'}
    with pytest.raises(ValueError, match='unknown or repeated'):
        client.annotate_segment({'prompt': 'original prompt'})
    write_atomic(tmp_path / 'requests/call_002/messages.json', [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'original prompt'}]}])
    with pytest.raises(ValueError, match='duplicate'):
        ReplyReplay(tmp_path)


def test_rgb_prepare_request_export_evaluate_without_network(tmp_path):
    import av
    import numpy as np
    import yaml
    from action_benchmark.dataset import prepare, read
    from action_benchmark.__main__ import export
    from label import harness

    data = tmp_path / 'source'
    video = data / 'videos/episode_000000.mp4'
    video.parent.mkdir(parents=True)
    with av.open(str(video), 'w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    action = {'subject': 'left_effector', 'skill': 'pick', 'slots': {}, 'text': 'pick cup'}
    schema = {'annotation_spec_version': 'skill_spec_v1', 'skills': [{'actions': [action]}],
              'episode_annotation': [{'id': 0, 'frame_count': 20, 'subtasks': [
                  {'start_frame': 0, 'end_frame': 20, 'primary_action': action, 'description': 'SECRET_LABEL'}]}]}
    write_atomic(data / 'lite_annotations/annotation_schema_v1.json', schema)
    write_atomic(data / 'meta/info.json', {'fps': 10, 'video_path': 'videos/episode_{episode_index:06d}.mp4',
                                         'features': {'head': {'dtype': 'video', 'shape': [48, 64, 3]}}})
    (data / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': 20, 'tasks': ['SECRET_ORDER']}))
    vocab = tmp_path / 'vocab.yaml'
    vocab.write_text(yaml.safe_dump({'version': 'skill_spec_v1', 'skills': {'pick': {'completion_rule': 'lift'}}}))
    job = tmp_path / 'job'
    manifest = prepare(data, vocab, job)
    ep = job / 'episodes/episode_000000'
    receipt = job / 'labels/episode_000000.json'
    record = harness.label_episode(ep, receipt, model='openai/gpt-6-astra', reasoning='medium', api_key='dry-run',
                                   max_tokens=64000, timeout=6060, cell_w=448, dry_run=True)
    assert 'SECRET' not in record['prompt_text']
    assert 'constrained_subtasks' in record['prompt_text']
    with pytest.raises(ValueError, match='blind whole-episode'):
        harness.label_episode(ep, receipt, model='openai/gpt-6-astra', reasoning='medium', api_key='dry-run',
                              max_tokens=64000, timeout=6060, cell_w=448, dry_run=True, dataset_note='SECRET_ORDER')
    reply = {'task_summary': 'pick cup', 'timeline': [{'start_s': 0, 'end_s': 2, 'action': 'pick'}], 'tasks': [],
             'constrained_subtasks': [{'start_frame': 0, 'end_frame': 20,
                                      'action_id': manifest['catalog']['actions'][0]['action_id']}]}
    record['labels'], record['parse_ok'] = harness.read_reply(json.dumps(reply), record)
    assert record['parse_ok'], record['labels']
    write_atomic(receipt, record)
    predictions = export(job, receipt.parent)
    assert evaluate(job, predictions)['summary']['macro_segment_f1_0.5'] == 1
    del record['input_identity']
    write_atomic(receipt, record)
    with pytest.raises(ValueError, match='missing benchmark input proof'):
        export(job, receipt.parent)
    schema['episode_annotation'][0]['frame_count'] = 99
    write_atomic(data / 'lite_annotations/annotation_schema_v1.json', schema)
    with pytest.raises(ValueError, match='frame count mismatch'):
        prepare(data, vocab, tmp_path / 'bad-job')
    assert not (tmp_path / 'bad-job').exists()
