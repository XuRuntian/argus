"""Replay blinded agent replies through native parsers; never dispatch an API call."""
from __future__ import annotations

from pathlib import Path

from action_benchmark.dataset import read
from label.atomic import write_atomic

MODEL = 'codex-agent-substitute'


class ReplyReplay:
    """Bind each saved reply to the exact native prompt, not call order."""

    model_name = MODEL

    def __init__(self, root):
        self.replies = {}
        self.used = set()
        for request in sorted((root / 'requests').glob('call_*/messages.json')):
            messages = read(request)
            prompt = messages[-1]['content'][0]['text']
            if prompt in self.replies:
                raise ValueError('duplicate replay prompt')
            self.replies[prompt] = read(root / 'replies' / (request.parent.name + '.json'))

    def annotate_segment(self, request):
        prompt = request['prompt']
        if prompt not in self.replies or prompt in self.used:
            raise ValueError('unknown or repeated replay prompt')
        self.used.add(prompt)
        return self.replies[prompt]


def replay_argus(root):
    from label import harness
    from action_benchmark.__main__ import export, evaluate

    root = Path(root).resolve()
    out = root / 'agent-substitute/argus'
    job = root / 'argus-job'
    record = read(root / 'argus-v2/dry-run.json')
    record.pop('dry_run', None)
    record.update(model=MODEL, model_served=MODEL, reasoning_effort=None, provider_name='agent substitution',
                  transport='blinded_agent_replay_v1', finish_reason='stop', usage=None,
                  response_text=(out / 'reply.json').read_text())
    record['agent_substitution'] = {
        'original_requested_model': 'qwen3.6-plus',
        'note': 'Original dry-run request policy retained; not a DashScope model result.',
        'request': str(out / 'requests/call_001/messages.json'),
    }
    record['config']['request_images']['dispatch_status'] = 'response_received'
    record['labels'], record['parse_ok'] = harness.read_reply(record['response_text'], record)
    ep = read(job / 'manifest.json')['episodes'][0]
    write_atomic(out / 'labels' / (ep['directory'] + '.json'), record)
    problems = harness.annotation_problems(record)
    write_atomic(out / 'validation.json', {'parse_ok': record['parse_ok'], 'problems': problems})
    if problems or not record['parse_ok']:
        raise ValueError(f'agent reply failed native validation: {problems}')
    predictions = export(job, out / 'labels')
    predictions['predictor'].update(model=MODEL, test_kind='agent_substitution')
    for episode in predictions['episodes']:
        episode['provenance']['agent_substitution'] = record['agent_substitution']
    write_atomic(out / 'predictions.json', predictions)
    write_atomic(out / 'metrics.json', evaluate(job, predictions))


def replay_alignment(root):
    from robo_subtask_alignment import FullFlowConfig, VlmAnnotationConfig, load_task_annotation
    from robo_subtask_alignment.reports import QualityGateReport
    from robo_subtask_alignment.vlm_annotation.pipeline import run_vlm_annotation
    from robo_subtask_alignment.vlm_annotation.phase_policy import load_builtin_phase_policy
    from robo_subtask_alignment.vlm_annotation.vocabulary import load_skill_phase_vocabulary
    from label.action_contract import action_id
    from action_benchmark.experiments.evaluation import evaluate

    root = Path(root).resolve()
    source = root / 'alignment-v2'
    out = root / 'agent-substitute/alignment'
    config = FullFlowConfig.model_validate(read(source / 'config.json'))
    quality = QualityGateReport.model_validate(read(source / 'workflow/quality_gate/report.json'))
    client = ReplyReplay(out)
    report = run_vlm_annotation(VlmAnnotationConfig(
        dataset_dir=config.dataset_dir,
        subtasks_file=quality.accepted_subtasks_file,
        output_dir=out / 'workflow',
        source_annotations_file=quality.source_annotations_file,
        exemplar_episode_indices=quality.exemplar_episode_indices,
        episodes=[1], task_id=quality.task_id, model=MODEL,
        max_images_per_segment=6, media_presentation=config.vlm.media_presentation,
        max_repair_attempts=0,
    ), client=client, phase_policy=load_builtin_phase_policy(load_skill_phase_vocabulary(None)))
    write_atomic(out / 'report.json', report.model_dump(mode='json'))
    if report.status != 'success' or report.failure_count or len(client.used) != len(client.replies):
        raise ValueError('native replay incomplete; inspect report before evaluation')
    annotation = load_task_annotation(report.episode_annotations_file)
    manifest = read(root / 'argus-job/manifest.json')
    contract = manifest['catalog']
    known = {row['action_id'] for row in contract['actions']}
    ep = manifest['episodes'][0]
    predicted = next(row for row in annotation.episode_annotation if row.id == ep['episode_id'])
    segments = []
    for row in predicted.subtasks:
        action = row.primary_action.model_dump(mode='json')
        key = action_id(action)
        segments.append({'start_frame': row.start_frame, 'end_frame': row.end_frame,
                         'action_id': key if key in known else None,
                         'reason': None if key in known else 'output template not in benchmark catalog',
                         'primary_action': action})
    predictions = {'version': 1, 'dataset_id': manifest['dataset_id'],
                   'catalog_sha256': contract['catalog_sha256'],
                   'predictor': {'name': 'robo-subtask-alignment', 'model': MODEL,
                                 'test_kind': 'agent_substitution',
                                 'input_modality': 'RGB + state + exemplar',
                                 'exemplar_episodes': [0], 'vlm_status': report.status,
                                 'successful_vlm_segments': report.annotation_count},
                   'episodes': [{**ep, 'subtasks': segments}]}
    write_atomic(out / 'predictions.json', predictions)
    write_atomic(out / 'metrics.json', evaluate(root / 'argus-job', predictions))
