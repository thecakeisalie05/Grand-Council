"""Auditable multi-agent deliberation. Each visible speech corresponds to an actual backend call."""
import json
import re
import threading
from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def parse_report(text):
    raw = (text or '').strip()
    candidates = [raw]
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.S | re.I)
    if match:
        candidates.insert(0, match.group(1))
    start, end = raw.find('{'), raw.rfind('}')
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and 'findings' in result:
                return result
        except (ValueError, TypeError):
            pass
    return {'executive_summary': raw, 'findings': [], 'agreement': [],
            'dissent': [], 'open_questions': [], 'next_steps': [],
            'parse_warning': 'Chair did not return parseable structured JSON; raw synthesis preserved.'}


def brief(text, n=1600):
    if len(text) <= n:
        return text
    return text[:n] + f'\n[… {len(text) - n} characters omitted from this review excerpt; full message is stored.]'


# A profile allocates different budgets to different kinds of deliberation. Each
# budget applies to ONE generation; optional continuations are separately bounded.
RESPONSE_BUDGETS = {
    'concise': {'proposal': 768, 'critique': 512, 'revision': 768, 'synthesis': 1536,
                'digestion': 512, 'repair': 1536},
    'balanced': {'proposal': 1536, 'critique': 896, 'revision': 1280, 'synthesis': 3072,
                 'digestion': 768, 'repair': 2048},
    'thorough': {'proposal': 2560, 'critique': 1536, 'revision': 2048, 'synthesis': 4096,
                 'digestion': 1024, 'repair': 3072},
}


def phase_kind(phase):
    if phase == 'Independent proposals':
        return 'proposal'
    if 'critiques' in phase:
        return 'critique'
    if 'revisions' in phase:
        return 'revision'
    if phase == 'Final synthesis':
        return 'synthesis'
    if phase == 'Report JSON repair':
        return 'repair'
    return 'digestion'


def merge_continuation(previous, following):
    """Remove exact overlaps when a chat model repeats the end of its prior answer."""
    if not following:
        return previous
    if previous.endswith(following):
        return previous
    for size in range(min(len(previous), len(following), 600), 11, -1):
        if previous.endswith(following[:size]):
            return previous + following[size:]
    return previous + following


class Cancelled(Exception):
    pass


class CouncilRunner:
    def __init__(self, store, models, engine):
        self.store = store
        self.models = models
        self.engine = engine
        self.guard = threading.Lock()
        self.active = None
        self.cancellation = set()

    def launch(self, run, executor):
        with self.guard:
            if self.active:
                raise RuntimeError('Another council is active. Finish or cancel it first.')
            self.active = run['id']
        try:
            self.store.put_run(run)
            executor.submit(self._execute, run)
        except Exception:
            with self.guard:
                self.active = None
            raise

    def cancel(self, run_id):
        with self.guard:
            if self.active != run_id:
                return False
            self.cancellation.add(run_id)
            return True

    def _check(self, run):
        with self.guard:
            if run['id'] in self.cancellation:
                raise Cancelled('Cancelled by user after the current inference call.')

    def _save(self, run, phase=None):
        if phase:
            run['phase'] = phase
        run['updated_at'] = now()
        self.store.put_run(run)

    def _say(self, run, phase, agent, prompt, max_tokens=None, temperature=None):
        self._check(run)
        model_cfg = self.models[agent['model_id']]
        context = [
            {'role': 'system', 'content': (
                f"You are {agent['name']}, a member of a technical design review council. "
                f"Your mandate: {agent['role']}. "
                'Be rigorous, concrete and independent. Explicitly distinguish known facts, '
                'inferences and assumptions. Do not invent tests, sources or consensus. '
                'Challenge hidden assumptions and propose actionable alternatives. '
                'You are one independent model call; never pretend you have seen evidence '
                'not supplied in the conversation.')},
            {'role': 'user', 'content': prompt},
        ]
        kind = phase_kind(phase)
        profile = run.get('response_detail', 'balanced')
        budget = RESPONSE_BUDGETS.get(profile, RESPONSE_BUDGETS['balanced'])[kind]
        if max_tokens is not None:
            budget = int(max_tokens)
        elif agent.get('max_tokens') is not None:  # Legacy per-agent limit (old clients).
            budget = int(agent['max_tokens'])
        budget = max(128, min(4096, budget))
        temp = temperature if temperature is not None else agent.get('temperature', .45)
        permitted = int(run.get('max_continuations', 1)) if run.get('auto_continue', True) else 0
        permitted = max(0, min(2, permitted))
        text = ''
        attempts = []
        status = 'unknown'
        problem = None
        self._save(run, f'{phase} · {agent["name"]}')
        for part in range(permitted + 1):
            self._check(run)
            if part:
                # Preserve the original specification and the accumulated response.
                # This is a fresh chat turn, not a guarantee of exact token-level continuation.
                # Providers with a small context may reject it; the partial text is retained.
                continuation = (context + [
                    {'role': 'assistant', 'content': text},
                    {'role': 'user', 'content': (
                        'Your previous response stopped because of the generation length limit. '
                        'Continue exactly where it stopped, without restarting, repeating any prior '
                        'text, or adding a new introduction. Finish the original task. '
                        'If the response was JSON, output only the remaining JSON characters.')},
                ])
                self._save(run, f'{phase} · {agent["name"]} · continuation {part}/{permitted}')
            else:
                continuation = context
            try:
                raw = self.engine.generate(model_cfg, continuation, max_tokens=budget, temperature=temp)
            except Exception as exc:
                if not part:
                    raise  # Never conceal initial inference/configuration errors.
                problem = f'Continuation {part} failed: {type(exc).__name__}: {exc}'
                status = 'partial'
                break
            # Support old/injected backends returning plain strings (finish reason unknown).
            if isinstance(raw, str):
                raw = {'text': raw, 'finish_reason': 'unknown', 'output_tokens': None}
            fragment = raw.get('text') or ''
            reason = str(raw.get('finish_reason') or 'unknown').lower()
            if reason in ('max_tokens', 'max_new_tokens', 'max_output_tokens'):
                reason = 'length'
            attempts.append({'finish_reason': reason, 'output_tokens': raw.get('output_tokens'),
                             'applied_max_tokens': raw.get('applied_max_tokens', budget)})
            merged = merge_continuation(text, fragment)
            if part and (not fragment.strip() or merged == text):
                problem = 'Continuation returned no new content.'
                status = 'partial'
                break
            text = merged
            if reason in ('stop', 'eos_token', 'end_turn'):
                status = 'complete'
                break
            if reason != 'length':
                status = 'unknown' if reason == 'unknown' else 'partial'
                if reason not in ('unknown',):
                    problem = f'Inference stopped with reason: {reason}.'
                break
            status = 'partial'  # Length limit exhausted; next iteration may finish it.
        if status == 'partial' and not problem:
            problem = (f'Output still reached the per-call limit (up to {budget} tokens, context-adjusted if needed) '
                       f'after {len(attempts)} generation call(s).')
        entry = {'id': 'm' + uuid4().hex[:10], 'phase': phase,
                 'agent_id': agent['id'], 'agent': agent['name'],
                 'model_id': agent['model_id'], 'model_name': model_cfg['label'],
                 'text': text, 'created_at': now(),
                 'completion_status': status,
                 'finish_reason': attempts[-1]['finish_reason'] if attempts else 'error',
                 'generation_calls': len(attempts), 'per_call_budget': budget,
                 'output_tokens': (sum(x['output_tokens'] for x in attempts)
                                   if attempts and all(isinstance(x['output_tokens'], int) for x in attempts)
                                   else None),
                 'generation_attempts': attempts}
        if problem:
            entry['completion_warning'] = problem
            run['warnings'].append(f'{agent["name"]} · {phase}: {problem} Review message {entry["id"]}.')
        run['messages'].append(entry)
        self._save(run)
        return entry

    def _condense(self, run, chair, text, max_chars):
        """Covers every part of the full specification via hierarchical model summaries."""
        if len(text) <= max_chars:
            return text
        original_len = len(text)
        source = text
        level = 0
        while len(source) > max_chars:
            level += 1
            if level > 9:
                raise RuntimeError('Specification could not be condensed into the configured context budget.')
            size = max(2500, max_chars // 2)
            chunks = [source[i:i + size] for i in range(0, len(source), size)]
            digests = []
            for i, chunk in enumerate(chunks):
                self._check(run)
                prompt = (f'Hierarchical spec digestion, level {level}, chunk {i+1}/{len(chunks)}. '
                          'Summarize ALL substantive requirements, constraints, acceptance criteria, '
                          'nonfunctional requirements, interfaces, dependencies, and decisions in this chunk. '
                          'Do not add inventions. Preserve technical identifiers. '
                          'Clearly note any ambiguous or conflicting requirements. '
                          'Be concise and information dense.\n\n' + chunk)
                message = self._say(run, f'Spec digestion L{level} {i+1}/{len(chunks)}', chair,
                                    prompt, temperature=0)
                digests.append(message['text'])
            new_source = '\n\n'.join(f'[DIGEST {i+1}]\n{d}' for i, d in enumerate(digests))
            if len(new_source) >= len(source) and len(source) > max_chars:
                raise RuntimeError('Digest did not reduce spec size. Use a higher input budget or a better summarization model.')
            source = new_source
        run['warnings'].append(
            f'Specification was {original_len:,} characters. Every chunk was read by the '
            f'chair model, but agents debated a lossy {len(source):,}-character hierarchical digest. '
            'Inspect the original spec for details the digest may have missed.')
        self._save(run)
        return source

    def _execute(self, run):
        try:
            run['status'] = 'running'
            self._save(run, 'Preparing session')
            agents = run['agents']
            chair = next(a for a in agents if a['id'] == run['chair_id'])
            spec = self._condense(run, chair, run['spec'], run['input_budget'])
            rules = '\n'.join(f'- {x}' for x in run.get('criteria', []) if x.strip())
            spec_context = f'TITLE: {run["title"]}\n\nSPECIFICATION:\n{spec}\n\nREVIEW CRITERIA:\n{rules}'
            last = {}

            # Independent initial proposals. The order is stable and not presented as parallel thought.
            for agent in agents:
                prompt = (spec_context + '\n\nSTAGE: INDEPENDENT REVIEW. You have not seen any other agent output. '
                          'Provide: (1) essential interpretation, (2) 3-8 prioritized suggestions, '
                          '(3) failure modes and trade-offs, (4) uncertainties, (5) validation plan. '
                          'Cite requirement names or exact snippets where possible. Avoid false confidence.')
                last[agent['id']] = self._say(run, 'Independent proposals', agent, prompt)

            for round_idx in range(1, run['rounds'] + 1):
                # Each member receives all others' prior outputs, not their own, and responds independently.
                critiques = {}
                for agent in agents:
                    peer_context = '\n\n'.join(
                        f'[{other["name"]} | message {last[other["id"]]["id"]}]\n'
                        + brief(last[other['id']]['text'], 2200)
                        for other in agents if other['id'] != agent['id'])
                    prompt = (spec_context + f'\n\nDEBATE ROUND {round_idx}/{run["rounds"]}. '
                              'Challenge and engage with the following peer positions. '
                              'Identify specific agreements, contradictions, missing requirements, '
                              'unsupported claims, viable fixes, and remaining disagreements. '
                              'Do not merely restate your own proposal. Refer to peer names/message IDs. '
                              '\n\nPEER POSITIONS:\n' + peer_context)
                    critiques[agent['id']] = self._say(run, f'Round {round_idx} · critiques', agent, prompt)
                revised = {}
                for agent in agents:
                    responses = '\n\n'.join(
                        f'[{other["name"]} | message {critiques[other["id"]]["id"]}]\n'
                        + brief(critiques[other['id']]['text'], 1800)
                        for other in agents)
                    prompt = (spec_context + '\n\nYOUR PREVIOUS POSITION:\n'
                              + brief(last[agent['id']]['text'], 2800)
                              + f'\n\nROUND {round_idx} CRITIQUES:\n' + responses
                              + '\n\nRevise your recommendations in light of the critique. '
                              'Explicitly state what you changed and why, what you reject and why, '
                              'and your remaining strongest recommendations. Do not manufacture agreement.')
                    revised[agent['id']] = self._say(run, f'Round {round_idx} · revisions', agent, prompt)
                last = revised

            dialogue = '\n\n'.join(
                f'[{agent["name"]} | message {last[agent["id"]]["id"]}]\n'
                + brief(last[agent['id']]['text'], 3800) for agent in agents)
            prompt = (spec_context + '\n\nFINAL POSITIONS:\n' + dialogue
                      + '\n\nSYNTHESIZE AS COUNCIL CHAIR. Do not force consensus. '
                      'Support important recommendations with message IDs (e.g. "m0123456789") '
                      'in each finding.sources array. Where there is no proof from tests say so. '
                      'RETURN VALID JSON ONLY, with shape: '
                      '{"executive_summary":"string",'
                      '"findings":[{"title":"string","severity":"critical|high|medium|low",'
                      '"recommendation":"string","rationale":"string","sources":["message-id"]}],'
                      '"agreement":["string"],"dissent":["string"],'
                      '"open_questions":["string"],"next_steps":["string"]}. '
                      'Findings should prioritize actionable improvements and architecture tradeoffs.')
            synthesis = self._say(run, 'Final synthesis', chair, prompt,
                                  temperature=0.2)
            report = parse_report(synthesis['text'])
            if report.get('parse_warning'):
                repair_prompt = ('Reformat this preceding chair synthesis into VALID JSON ONLY. '
                                 'Do not add findings, facts or evidence. Preserve uncertainty. '
                                 'Required top-level keys: executive_summary, findings, agreement, '
                                 'dissent, open_questions, next_steps. Each finding should have '
                                 'title, severity, rationale, recommendation, sources.\n\n'
                                 + brief(synthesis['text'], 5200))
                repaired = self._say(run, 'Report JSON repair', chair, repair_prompt,
                                     temperature=0)
                maybe = parse_report(repaired['text'])
                if not maybe.get('parse_warning'):
                    report = maybe
                else:
                    run['warnings'].append('Chair could not return structured JSON; inspect the raw synthesis.')
            self._check(run)
            run['report'] = report
            run['status'] = 'completed'
            self._save(run, 'Complete')
        except Cancelled as exc:
            run['status'] = 'cancelled'
            run['error'] = str(exc)
            self._save(run, 'Cancelled')
        except Exception as exc:
            run['status'] = 'failed'
            run['error'] = f'{type(exc).__name__}: {exc}'
            self._save(run, 'Error')
        finally:
            with self.guard:
                self.cancellation.discard(run['id'])
                if self.active == run['id']:
                    self.active = None


def to_markdown(run):
    report = run.get('report') or {}
    out = [f'# Grand Council: {run["title"]}',
           f'\nStatus: **{run["status"]}** | {run["created_at"]} | {run["rounds"]} debate round(s)',
           '\n## Executive summary\n', report.get('executive_summary', 'No final report yet.'),
           '\n## Findings\n']
    for i, finding in enumerate(report.get('findings', []), 1):
        out += [f'### {i}. {finding.get("title", "Finding")} ({finding.get("severity", "unrated")})',
                finding.get('rationale', ''), '\n**Recommendation:** ' + finding.get('recommendation', ''),
                '\n**Message references:** ' + ', '.join(finding.get('sources', []))]
    for title, key in [('Agreement', 'agreement'), ('Dissent', 'dissent'),
                       ('Open questions', 'open_questions'), ('Next steps', 'next_steps')]:
        out += ['\n## ' + title]
        if report.get(key):
            out.extend('- ' + (x if isinstance(x, str) else str(x)) for x in report[key])
        else:
            out.append('- None recorded')
    if run.get('warnings'):
        out += ['\n## Context warnings'] + ['- ' + w for w in run['warnings']]
    out += ['\n## Original specification\n', run['spec'], '\n## Full deliberation transcript\n']
    for m in run.get('messages', []):
        status = m.get('completion_status', 'not recorded')
        calls = m.get('generation_calls', 'not recorded')
        out += [f'### {m["agent"]} — {m["phase"]} ({m["id"]})',
                f'*Model: {m["model_name"]} · {m["created_at"]} · Completion: {status} · Generations: {calls}*\n',
                ('**Generation warning:** ' + m['completion_warning'] if m.get('completion_warning') else ''),
                m['text'], '']
    return '\n'.join(out)
