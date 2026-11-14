"""Exact source provenance for model-planned host interfaces, independent of protocol.

This checks origin and bounded expansion, not the semantic correctness of an API
description. Compilation and runtime evaluation remain separate requirements.
"""
import hashlib
import json
from pathlib import Path
import re

GROUNDING_POLICY = 'host_source_inventory_v1'
HOST_PAGE_SIZE = 4
HOST_PLAN_MAX_CALLS = 24  # Interface + inventory + detail pages + four guidance units.


class HostEvidenceError(ValueError):
    """Rejected evidence plus exact source locations for the existing repair call."""
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def contains_identifier(text, name):
    return bool(name and re.search(r'(?<![A-Za-z0-9_])' + re.escape(name)
                                  + r'(?![A-Za-z0-9_])', text))


def documented_symbol(name, source_name):
    if name == source_name:
        return True
    # Plugin-local declarations may generalize one documented example variable
    # to a protocol placeholder. Arbitrary renaming/numbered suffixes may not.
    placeholders = list(re.finditer(r'\{(?:protocol|Protocol)\}', name))
    if len(placeholders) != 1:
        return False
    match = placeholders[0]
    pattern = re.escape(name[:match.start()]) + r'[A-Za-z_][A-Za-z0-9_]*' + re.escape(name[match.end():])
    return bool(re.fullmatch(pattern, source_name))


class HostSourceCorpus:
    def __init__(self, docs):
        if not docs or any(not chunk.content.strip() for chunk in docs):
            raise ValueError('Host grounding requires nonempty original source documents')
        self.documents = {}
        for index, chunk in enumerate(docs):
            digest = hashlib.sha256(chunk.content.encode('utf-8')).hexdigest()
            source_id = f'doc-{index}-{digest[:16]}'
            self.documents[source_id] = dict(source_id=source_id, sha256=digest,
                name=Path(chunk.source).name, kind=chunk.kind, content=chunk.content)
        self.manifest = [{k: v for k, v in doc.items() if k != 'content'}
                         for doc in self.documents.values()]
        self.sha256 = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()

    def render(self):
        return '\n\n'.join(f"SOURCE {doc['source_id']} | {doc['name']} | {doc['kind']}\n"
                            + doc['content'] + f"\nEND SOURCE {doc['source_id']}"
                            for doc in self.documents.values())

    def validate_inventory(self, value):
        planned = {}
        for kind, key in (('symbol', 'symbol_names'), ('api', 'api_names')):
            names = value[key]
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError('Empty inventory name')
            if len(names) != len(set(names)):
                raise ValueError('Duplicate inventory name')
            planned.update({(kind, name): None for name in names})
        if not value['api_names']:
            raise ValueError('Inventory requires documented host APIs')
        evidence = []
        for row in value['source_evidence']:
            key = (row['kind'], row['name'])
            if key not in planned or planned[key] is not None:
                raise ValueError('Extra or duplicate source evidence for inventory entry')
            doc = self.documents.get(row['source_id'])
            if doc is None:
                raise ValueError('Unknown original host source_id: ' + row['source_id'])
            quote, source_name = row['quote'], row['source_name']
            if not quote.strip() or quote not in doc['content']:
                candidates = []
                offset = 0
                for line_number, line in enumerate(doc['content'].splitlines(keepends=True), 1):
                    if contains_identifier(line, source_name) and len(line) <= 1200:
                        candidates.append(dict(source_id=row['source_id'], line=line_number,
                            quote_offset=offset, quote=line.rstrip('\r\n'), source_sha256=doc['sha256']))
                    offset += len(line)
                raise HostEvidenceError('Source quote is not verbatim original host evidence: ' + row['name'],
                    dict(category='host_source_quote', name=row['name'], source_name=source_name,
                         candidates=candidates[:3], candidate_count=len(candidates),
                         semantic_correctness_verified=False,
                         action='Select an appropriate verbatim contiguous quote from the original source. '
                         'Candidates are identifier locations, not proof of API semantics. Preserve exact '
                         'whitespace within the quote; do not paraphrase or join separate passages. '
                         'Revalidate the entire inventory after replacing the failed evidence.'))
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*', source_name):
                raise ValueError('Invalid documented source identifier')
            if not contains_identifier(quote, source_name):
                raise ValueError('Source quote omits the complete identifier: ' + source_name)
            if (row['kind'] == 'api' and row['name'] != source_name) or (
                    row['kind'] == 'symbol' and not documented_symbol(row['name'], source_name)):
                raise ValueError('Inventory name is not the documented identifier/template: ' + row['name'])
            offset = doc['content'].index(quote)
            verified = {**row, 'source_sha256': doc['sha256'], 'quote_offset': offset,
                        'line': doc['content'].count('\n', 0, offset) + 1}
            planned[key] = verified
            evidence.append(verified)
        missing = [name for (kind, name), row in planned.items() if row is None]
        if missing:
            raise ValueError('Missing original source evidence: ' + ', '.join(missing[:8]))
        planned_calls = 6 + sum((len(value[key]) + HOST_PAGE_SIZE - 1) // HOST_PAGE_SIZE
                                for key in ('symbol_names', 'api_names'))
        if planned_calls > HOST_PLAN_MAX_CALLS:
            raise ValueError(f'Host inventory needs {planned_calls} calls, exceeding source-plan budget '
                             f'{HOST_PLAN_MAX_CALLS}. Return a minimal sufficient documented interface; '
                             'do not enumerate the entire SDK or numbered variants.')
        return dict(policy=GROUNDING_POLICY, corpus_sha256=self.sha256, sources=self.manifest,
                    planned_host_calls=planned_calls, max_host_calls=HOST_PLAN_MAX_CALLS,
                    evidence=evidence, origin_verified=True, semantic_correctness_verified=False)
