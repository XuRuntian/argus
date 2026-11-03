import copy
import json

import av
import numpy as np
import pytest
import yaml

from label import harness, subtasks
from label.atomic import write_atomic
from subtask_annotation.__main__ import DEFAULT_SPEC, export, prepare, read, rgb_files


@pytest.fixture
def contract():
    return subtasks.bind(yaml.safe_load(DEFAULT_SPEC.read_text()), 20, 10)


def segment(**changes):
    return {'start_frame': 0, 'end_frame': 20, 'action': 'hold',
            'subject': 'right_effector', 'object': 'cup', **changes}


def reply(rows):
    return json.dumps({'task_summary': 'hold cup', 'tasks': [],
                       'timeline': [{'start_s': 0, 'end_s': 2, 'action': 'hold cup'}],
                       'subtasks': rows})


def change_point(**changes):
    return {k: v for k, v in segment(**changes).items() if k != 'end_frame'}


def test_change_points_materialize_without_mutation_and_reparse(contract):
    rows = [change_point(), change_point(start_frame=8),
            change_point(start_frame=12, action=None, subject='unknown', object=None),
            change_point(start_frame=17, action='none', subject='unknown', object=None)]
    original = copy.deepcopy(rows)
    request = {'subtask_contract': contract, 'annotation_policy': {'rig': 'teleop_arms'}}
    labels, ok = harness.read_reply(reply(rows), request)
    assert ok
    assert rows == original
    assert labels['subtasks'] == [{**row, 'end_frame': end}
                                  for row, end in zip(rows, [8, 12, 17, 20])]
    checked, ok = harness.read_reply(json.dumps(labels), request)
    assert ok and checked == labels
    assert subtasks.materialize([change_point()], contract) == [segment()]


@pytest.mark.parametrize('rows', [
    [change_point(start_frame=1)], [change_point(start_frame=-1)],
    [change_point(start_frame=True)], [change_point(start_frame=0.0)],
    [change_point(start_frame='0')],
    [change_point(), change_point(start_frame=20)],
    [change_point(), change_point(start_frame=21)],
    [change_point(), change_point()],
    [change_point(), change_point(start_frame=10), change_point(start_frame=5)],
    [change_point(), segment(start_frame=10)],
    [change_point(action='none')], [change_point(subject=None)],
    [change_point(extra='unexpected')],
])
def test_bad_change_points_remain_unchanged_and_incomplete(contract, rows):
    labels, ok = harness.read_reply(reply(rows), {'subtask_contract': contract})
    assert not ok
    assert labels['subtasks'] == rows
    assert 'subtask contract:' in labels['_incomplete'][0]


def test_prompt_contains_action_boundaries_but_no_internal_contract(contract):
    text = subtasks.prompt(contract)
    assert {'hold', 'hand_over', 'return_to_initial_pose'} <= set(contract['spec']['actions'])
    assert 'Start:' in text and 'End:' in text
    assert 'Do not output end_frame' in text
    assert 'four fields' in text
    for forbidden in ('allowed_phases', 'phases', 'vocabulary_sha256', 'spec_sha256', 'required_arguments'):
        assert forbidden not in text
    assert contract['spec_sha256'] not in text
    changed = copy.deepcopy(contract)
    changed['spec']['actions']['hold']['end'] = 'changed'
    with pytest.raises(ValueError, match='digest'):
        subtasks.validate([segment()], changed)


@pytest.mark.parametrize('rows', [[], {'skills': {}, 'phases': {}},
    [{'skills': {}, 'phases': {}, 'vocabulary_sha256': 'echo'}],
    [segment(start_frame=True)], [segment(end_frame=21)], [segment(start_frame=1)],
    [segment(end_frame=19)], [segment(end_frame=12), segment(start_frame=10)],
    *[segment_list for value in ('', ' ', 42, True, [], {}, ' name ')
      for segment_list in ([segment(action=value)], [segment(subject=value)])],
    [segment(subject=None)],
    [segment(object={'name': 'cup'})], [segment(phases=[])], [segment(action='none')]])
def test_invalid_or_echoed_output_is_retained_and_incomplete(contract, rows):
    labels, ok = harness.read_reply(reply(rows), {'subtask_contract': contract})
    assert not ok
    assert labels['subtasks'] == rows
    assert 'subtask contract:' in labels['_incomplete'][0]


def test_unknown_idle_and_repetitions_have_explicit_intervals(contract):
    rows = [segment(end_frame=5), segment(start_frame=5, end_frame=10),
            segment(start_frame=10, end_frame=15, action=None, subject='unknown', object=None),
            segment(start_frame=15, action='none', subject='unknown', object=None)]
    assert subtasks.validate(rows, contract) == rows
    # No opt-in: an extra malformed field must not change the old harness's completeness judgement.
    labels, ok = harness.read_reply(reply({'skills': 'not an annotation'}),
                                     {'annotation_policy': {'rig': 'teleop_arms'}})
    assert ok


def test_custom_action_and_subject_are_preserved_without_changing_vocabulary(contract):
    original = copy.deepcopy(contract)
    rows = [segment(action='peel', subject='human_hand', object='banana')]
    assert 'peel' not in contract['spec']['actions']
    assert 'human_hand' not in contract['spec']['subjects']
    labels, ok = harness.read_reply(reply(rows),
        {'subtask_contract': contract, 'annotation_policy': {'rig': 'teleop_arms'}})
    assert ok
    assert labels['subtasks'] == rows
    assert contract == original
    text = subtasks.prompt(contract)
    assert 'not exhaustive restrictions' in text
    assert 'Do not invent synonyms' in text
    assert 'For a freely named action' in text


def test_closed_vocabulary_rejects_new_actions_but_allows_subjects_and_unknown(contract):
    closed = {**contract, 'action_vocabulary': 'closed'}
    request = {'subtask_contract': closed, 'annotation_policy': {'rig': 'teleop_arms'}}
    for action in ('hold', None, 'none'):
        row = change_point(action=action, subject='human_hand', object=None)
        labels, ok = harness.read_reply(reply([row]), request)
        assert ok, labels
        assert labels['subtasks'] == [{**row, 'end_frame': 20}]
    for row in (change_point(action='peel'), segment(action='peel')):
        labels, ok = harness.read_reply(reply([row]), request)
        assert not ok
        assert labels['subtasks'] == [row]
        assert 'closed action vocabulary' in labels['_incomplete'][0]
    text = subtasks.prompt(closed)
    assert 'Action vocabulary is closed' in text
    assert 'Subject names remain extensible' in text
    assert 'For a freely named action' not in text
    assert 'not exhaustive restrictions' not in text


def test_legacy_contract_defaults_to_open(contract):
    legacy = {k: v for k, v in contract.items() if k != 'action_vocabulary'}
    assert subtasks.get_action_vocabulary(legacy) == 'open'
    assert subtasks.prompt(legacy) == subtasks.prompt(contract)
    assert subtasks.materialize([change_point(action='peel')], legacy) == [segment(action='peel')]


@pytest.mark.parametrize('mode', [None, '', 'invalid', [], True])
def test_invalid_vocabulary_mode_is_rejected(contract, mode):
    with pytest.raises(ValueError, match='action_vocabulary'):
        subtasks.validate_contract({**contract, 'action_vocabulary': mode})


def test_publication_does_not_rewrite_object_names():
    from label.wording import normalize_labels
    rows = [segment(object='the supplied window')]
    assert normalize_labels({'subtasks': rows})['subtasks'] == rows


def test_contract_is_in_request_code_identity_and_parts_are_rejected(contract):
    from pathlib import Path
    repository = Path(__file__).resolve().parents[1]
    assert repository / 'label/subtasks.py' in harness.request_implementation(repository)
    with pytest.raises(ValueError, match='whole episodes'):
        subtasks.require_context({'subtask_contract': contract, 'piece': {'of': 'episode_000000'}})


def source(tmp_path, *, channel_first=False):
    data = tmp_path / 'source'
    data.mkdir()
    with av.open(str(data / 'camera.mp4'), 'w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    info = {'fps': 10, 'video_path': 'camera.mp4', 'features': {
        'camera': {'dtype': 'video', 'shape': [3, 48, 64] if channel_first else [48, 64, 3]}}}
    write_atomic(data / 'meta/info.json', info)
    (data / 'meta/episodes.jsonl').write_text(json.dumps(
        {'episode_index': 0, 'length': 20, 'tasks': ['SECRET_TASK_ORDER']}))
    # No human answer file is required. Even a malformed one must not be read.
    (data / 'lite_annotations').mkdir()
    (data / 'lite_annotations/annotation_schema_v1.json').write_text('SECRET_ANSWER_NOT_JSON')
    return data


@pytest.mark.parametrize('row', [segment(), segment(action='peel', subject='human_hand', object='banana')])
@pytest.mark.parametrize('starts_only', [False, True])
def test_prepare_dry_run_mock_reply_export_and_identity(tmp_path, row, starts_only):
    data = source(tmp_path, channel_first=True)
    job = tmp_path / 'job'
    manifest = prepare(data, job)
    with pytest.raises(FileExistsError):
        prepare(data, job)
    assert read(job / 'manifest.json') == manifest
    ep = job / 'episodes/episode_000000'
    receipt = job / 'labels/episode_000000.json'
    options = dict(model='openai/gpt-6-astra', reasoning='medium', api_key='dry-run',
                   max_tokens=64000, timeout=6060, cell_w=448, dry_run=True)
    record = harness.label_episode(ep, receipt, **options)
    assert 'SECRET' not in record['prompt_text']
    assert 'ACTION DEFINITIONS' in record['prompt_text']
    with pytest.raises(ValueError, match='incomplete'):
        export(job, receipt.parent)
    with pytest.raises(ValueError, match='without task notes'):
        harness.label_episode(ep, receipt, **options, dataset_note='SECRET_TASK_ORDER')
    # Simulate a received response; no provider is contacted by this test.
    record.pop('dry_run', None)
    received = {k: v for k, v in row.items() if k != 'end_frame'} if starts_only else row
    record['labels'], record['parse_ok'] = harness.read_reply(reply([received]), record)
    assert record['parse_ok'], record['labels']
    write_atomic(receipt, record)
    output = export(job, receipt.parent)
    assert output['episodes'][0]['subtasks'] == [row]
    assert output['episodes'][0]['sampled_times_s'] == record['config']['timesteps_s']
    assert not ({'metrics', 'scores', 'phases'} & set(output))
    # A changed source invalidates export even though the annotation itself still parses.
    ctx = read(ep / 'context.json')
    ctx['changed'] = True
    write_atomic(ep / 'context.json', ctx)
    with pytest.raises(ValueError, match='changed or missing input'):
        export(job, receipt.parent)


def test_depth_is_not_rgb_and_failed_prepare_leaves_no_job(tmp_path):
    data = source(tmp_path)
    info = read(data / 'meta/info.json')
    info['features']['depth'] = {'dtype': 'video', 'shape': [48, 64, 3],
                                 'info': {'video.is_depth_map': True}}
    assert [v[0] for v in rgb_files(data, info, 0).values()] == ['camera']
    del info['features']['camera']
    write_atomic(data / 'meta/info.json', info)
    job = tmp_path / 'job'
    with pytest.raises(ValueError, match='no RGB'):
        prepare(data, job)
    assert not job.exists()


@pytest.mark.parametrize('mode', ['open', 'closed', 'legacy'])
def test_vocabulary_mode_roundtrip_export_and_mismatch(tmp_path, mode):
    from subtask_annotation.__main__ import main
    data = source(tmp_path)
    job = tmp_path / 'job'
    selected_mode = 'open' if mode == 'legacy' else mode
    main(['prepare', '--dataset', str(data), '--out', str(job),
          '--action-vocabulary', selected_mode])
    manifest = read(job / 'manifest.json')
    ep = job / 'episodes/episode_000000'
    ctx = read(ep / 'context.json')
    assert ctx['subtask_contract']['action_vocabulary'] == selected_mode
    assert manifest['action_vocabulary'] == selected_mode
    if mode == 'legacy':
        # Historical jobs and receipts omit the mode entirely.
        del manifest['action_vocabulary']
        del ctx['subtask_contract']['action_vocabulary']
        write_atomic(job / 'manifest.json', manifest)
        write_atomic(ep / 'context.json', ctx)
    receipt = job / 'labels/episode_000000.json'
    record = harness.label_episode(ep, receipt, model='openai/gpt-6-astra', reasoning='medium',
        api_key='dry-run', max_tokens=64000, timeout=6060, cell_w=448, dry_run=True)
    record.pop('dry_run')
    action = 'hold' if mode == 'closed' else 'peel'
    record['labels'], record['parse_ok'] = harness.read_reply(reply([change_point(action=action)]), record)
    assert record['parse_ok']
    write_atomic(receipt, record)
    output = export(job, receipt.parent)
    assert output['action_vocabulary'] == selected_mode
    assert output['episodes'][0]['subtasks'] == [segment(action=action)]
    # Altering only the manifest must not relabel a saved run as another mode.
    manifest['action_vocabulary'] = 'open' if selected_mode == 'closed' else 'closed'
    write_atomic(job / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='contract differs'):
        export(job, receipt.parent)


def test_non_cfr_and_unknown_selection_fail_without_artifacts(tmp_path):
    data = source(tmp_path)
    job = tmp_path / 'job'
    with pytest.raises(ValueError, match='selection'):
        prepare(data, job, selected=[5])
    assert not job.exists()
    info = read(data / 'meta/info.json')
    info['fps'] = 20
    write_atomic(data / 'meta/info.json', info)
    with pytest.raises(ValueError, match='CFR'):
        prepare(data, job)
    assert not job.exists()


def test_spec_rejects_phase_payload_and_cli_is_annotation_only(capsys):
    from subtask_annotation.__main__ import main
    spec = yaml.safe_load(DEFAULT_SPEC.read_text())
    spec['phases'] = {}
    with pytest.raises(ValueError, match='no phases'):
        subtasks.validate_spec(spec)
    with pytest.raises(SystemExit) as stopped:
        main(['--help'])
    assert stopped.value.code == 0
    assert '{prepare,export}' in capsys.readouterr().out
