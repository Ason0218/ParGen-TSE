"""Split specification documents at headings and select sections for PKG construction.

PDF text is extracted with pdftotext -layout. RFC text is read directly.
Repeated page headers and footers are removed. Numbered headings follow a
valid successor chain; RFC 792 uses its unnumbered column-zero headings.
Word documents use heading styles and preserve tables as rows of cell text.
Each section extends from its heading to the next heading.

The toc_llm mode sends each document's heading hierarchy and the target
protocol to the model. Selected sections retain their extracted text.
The unfiltered mode retains every section."""
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

METHOD = 'spec_text_sections_v1'
HEADING = re.compile(r'^(\s*)(\d+(?:\.\d+)*)\.?\s+(\S.*?)\s*$')
LEADERS = re.compile(r'(\.\s?){4,}')
# RFC 792 (icmp.pdf, the ICMP input) has no section numbers; these are its column-0 headings.
UNNUMBERED = {'icmp': ['Introduction', 'Message Formats', 'Destination Unreachable Message', 'Time Exceeded Message',
                       'Parameter Problem Message', 'Source Quench Message', 'Redirect Message',
                       'Echo or Echo Reply Message', 'Timestamp or Timestamp Reply Message',
                       'Information Request or Information Reply Message', 'Summary of Message Types', 'References']}


def pages_of(pdf):
    if pdf.suffix == '.txt':
        text = pdf.read_text(errors='replace')
    else:
        text = subprocess.run(['pdftotext', '-layout', '-enc', 'UTF-8', str(pdf), '-'],
                              check=True, capture_output=True, text=True).stdout
    pages = text.split('\f')
    return pages[:-1] if pages and not pages[-1].strip() else pages


def strip_running_lines(pages):
    """Drop lines that recur in the first/last three non-empty lines of many pages."""
    norm = lambda s: re.sub(r'\d+', '#', ' '.join(s.split()))
    edge = lambda lines: [i for i, l in enumerate(lines) if l.strip()][:3] + [i for i, l in enumerate(lines) if l.strip()][-3:]
    counts = Counter()
    split = [p.split('\n') for p in pages]
    for lines in split:
        counts.update({norm(lines[i]) for i in edge(lines)})
    running = {k for k, c in counts.items() if c >= max(3, 0.3 * len(pages)) and k}
    out = []
    for page_no, lines in enumerate(split, 1):
        drop = {i for i in edge(lines) if norm(lines[i]) in running}
        out.extend((page_no, l) for i, l in enumerate(lines) if i not in drop)
    return out


def successor(prev, cur):
    if prev is None:
        return cur == ('1',)
    if cur == prev + ('1',):
        return True
    for depth in range(len(prev), 0, -1):
        if cur == prev[:depth - 1] + (str(int(prev[depth - 1]) + 1),):
            return True
    return False


def headings(document, lines, column0=False):
    if document in UNNUMBERED:
        wanted, found = list(UNNUMBERED[document]), []
        for i, (_, line) in enumerate(lines):
            if wanted and line.rstrip() == wanted[0]:
                found.append((i, str(len(found) + 1), wanted.pop(0)))
        if wanted:
            raise ValueError(f'{document}: headings not found: {wanted}')
        return found
    # Candidates, then the longest chain of valid successors (so a table of contents that
    # already runs 1, 1.1, 2, ... is superseded by the body chain that follows it).
    cands = []
    for i, (_, line) in enumerate(lines):
        m = HEADING.match(line)
        # table rows have column gaps (3+ spaces) inside the text; headings do not
        if not m or LEADERS.search(line) or len(m.group(3).split()) > 16 or m.group(3).endswith(('.', ',', ';')) \
                or '   ' in m.group(3):
            continue
        if len(m.group(1)) > 4 and not m.group(3).isupper():  # indented body lines; centred top-level titles are upper case
            continue
        if column0 and m.group(1):  # RFC plain text: headings start in column 0, body lines are indented
            continue
        cands.append((i, tuple(m.group(2).split('.')), m.group(3)))
    # Longest chain by dynamic programming: a numbered body line ("4 The client may choose")
    # must not cut the real chain short, and on a tie the later chain (the body) wins.
    length, back = [0] * len(cands), [None] * len(cands)
    for k, (_, num, _) in enumerate(cands):
        if num == ('1',):
            length[k] = 1
        for j in range(k):
            if length[j] and successor(cands[j][1], num) and length[j] + 1 >= length[k] and length[j] + 1 > 1:
                length[k], back[k] = length[j] + 1, j
    if not any(length):
        raise ValueError(f'{document}: no numbered headings')
    k = max(range(len(cands)), key=lambda x: (length[x], x))
    best = []
    while k is not None:
        best.append(cands[k])
        k = back[k]
    return [(i, '.'.join(num), title) for i, num, title in reversed(best)]


W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
DOCX_HEADING = re.compile(r'^(?:Heading|H)([1-9])$')


def _docx_text(el):
    out = []
    for node in el.iter():
        if node.tag == W + 't':
            out.append(node.text or '')
        elif node.tag == W + 'tab':
            out.append('\t')
        elif node.tag in (W + 'br', W + 'cr'):
            out.append('\n')
    return ''.join(out)


def docx_sections(path):
    """(number, title, text) per heading of a Word specification, in document order."""
    import zipfile
    from xml.etree import ElementTree as ET
    body = ET.fromstring(zipfile.ZipFile(path).read('word/document.xml')).find(W + 'body')
    heads, current = [], None
    for el in body:
        if el.tag == W + 'p':
            style = el.find(f'{W}pPr/{W}pStyle')
            style = style.get(W + 'val') if style is not None else ''
            if style.upper().startswith('TOC') or style == 'TT':
                continue
            text = _docx_text(el)
            if DOCX_HEADING.match(style) and text.strip():
                number, _, title = text.strip().partition('\t')
                if not title:  # unnumbered heading (Foreword, Annex X ...)
                    number, title = text.strip(), text.strip()
                current = dict(number=' '.join(number.split()), title=' '.join(title.split()), lines=[text.strip()])
                heads.append(current)
            elif current is not None:
                current['lines'].append(text)
        elif el.tag == W + 'tbl' and current is not None:
            for row in el.iter(W + 'tr'):
                cells = [' '.join(_docx_text(p).split()) for p in row.findall(W + 'tc')]
                current['lines'].append(' | '.join(cells))
    return [(h['number'], h['title'], '\n'.join(h['lines']).rstrip()) for h in heads]


def input_documents(doc_dir):
    """The input documents in cut order: Word specifications, then PDFs and RFC texts."""
    doc_dir = Path(doc_dir)
    return [*sorted(doc_dir.glob('*.docx')), *sorted([*doc_dir.glob('*.pdf'), *doc_dir.glob('*.txt')])]


def cut_documents(doc_dir):
    """[{source_file, number, numbered, title, start_page, end_page, text}] in document order."""
    sections = []
    for doc in input_documents(doc_dir):
        if doc.suffix == '.docx':
            for number, title, text in docx_sections(doc):
                sections.append(dict(source_file=doc.stem, number=number, numbered=True, title=title,
                                     start_page=0, end_page=0, text=text))
            continue
        lines = strip_running_lines(pages_of(doc))
        heads = headings(doc.stem, lines, column0=doc.suffix == '.txt')
        for k, (i, number, title) in enumerate(heads):
            end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
            body = [l for _, l in lines[i:end]]
            while body and not body[-1].strip():
                body.pop()
            # numbered=False: the document has no section numbers (RFC 792); `number` is the
            # heading's position, for reference only, and must not be shown as a section number.
            sections.append(dict(source_file=doc.stem, number=number, numbered=doc.stem not in UNNUMBERED,
                                 title=' '.join(title.split()),
                                 start_page=lines[i][0], end_page=lines[end - 1][0],
                                 text='\n'.join(body)))
    return sections


def stage_b_sections(cut):
    """Stage B section records (the schema of a sections subset), each section whole."""
    return [dict(number=s['number'] if s.get('numbered', True) else '',
                 title=f"{s['number']} {s['title']}" if s.get('numbered', True) else s['title'],
                 source_file=s['source_file'], source_page=s['start_page'], end_page=s['end_page'],
                 extraction_method=METHOD, content=s['text'])
            for s in cut]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pdftotext_version():
    try:
        done = subprocess.run(['pdftotext', '-v'], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return (done.stderr or done.stdout).strip().splitlines()[0] if (done.stderr or done.stdout).strip() else None


TOC_PROMPT_VERSION = 'toc_select_v2'
TOC_SELECT_INSTRUCTIONS = """You are preparing the input for building a parser of the {protocol} protocol from its specification.
Below is the table of contents of one specification document ({document}): every section heading with its identifier, indented by nesting level. You see only the headings, not the text.
Judge relevance from the meaning of each heading and its place in the full hierarchy. A relevant heading need not contain words such as "packet", "format" or "field".

Select every section that a parser implementer has to read to
1. lay out the bytes of each message, packet, header or data unit the document defines (formats, fields, options, extensions, TLVs, information elements, encodings);
2. know what each field means, its units, and its allowed, reserved or coded values;
3. check a single received message for validity (length, presence and value rules that a receiver applies to one message).
Leave out sections that are only about the behaviour of endpoints over time (state machines, timers, procedures, session handling), configuration, management, security considerations, registry administration, references, acknowledgements and document history, unless the heading shows that the section defines the content or the validity of a message.
When unsure whether a section defines message content, include it. Parent sections can contain definitions before their subsections, so consider their own content too. Selecting a section does not include its subsections: list every section you want.

Reply with one JSON object and nothing else: {{"selected": ["<identifier exactly as listed>", ...]}}

Table of contents:
{toc}
"""
PROTOCOL_NAMES = {'bfd': 'BFD', 'dhcp': 'DHCP', 'icmp': 'ICMP', 'ipv4': 'IPv4', 'modbus': 'Modbus', 'tcp': 'TCP',
                  'dns': 'DNS', 'pfcp': 'PFCP'}


def toc_text(doc_sections):
    """The document's table of contents: one heading per line, indented by nesting level."""
    lines = []
    for s in doc_sections:
        level = s['number'].count('.') if s.get('numbered', True) else 0
        lines.append('  ' * level + f"{s['number']}  {s['title']}" if s['number'] != s['title'] else '  ' * level + s['title'])
    return '\n'.join(lines)


def toc_prompt(protocol, document, doc_sections):
    return TOC_SELECT_INSTRUCTIONS.format(protocol=PROTOCOL_NAMES.get(protocol, protocol), document=document,
                                          toc=toc_text(doc_sections))


def chat_completer(base_url, api_key, model, temperature=0.0, max_tokens=65536, timeout=3600, log_dir=None, attempts=3):
    """complete(prompt) -> (text, meta) on an OpenAI-compatible endpoint, JSON mode, T as given.

    A degenerate reply (empty content or cut off) is retried; with STEP2_DEGENERATE_RETRY_SAMPLING
    set, the retry is sampled (pargen/degenerate_retry.py), as in Stage B.
    """
    import requests
    from . import degenerate_retry

    def complete(prompt):
        built = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': temperature,
                 'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}}
        tries = []
        for _ in range(attempts):
            payload, sampled = degenerate_retry.apply(built)
            started = time.time()
            resp = requests.post(base_url, json=payload, timeout=(10, timeout),
                                 headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
            resp.raise_for_status()
            data = resp.json()
            degenerate_retry.note(built, data, sampled, log_dir=log_dir)
            choice = (data.get('choices') or [{}])[0]
            text = ((choice.get('message') or {}).get('content') or '').strip()
            tries.append(dict(finish_reason=choice.get('finish_reason'), usage=data.get('usage'), sampling=sampled,
                              seconds=round(time.time() - started, 1), empty=not text))
            if text and choice.get('finish_reason') != 'length':
                return text, dict(model=data.get('model') or model, attempts=tries)
        raise RuntimeError(f'table-of-contents selection: no usable reply after {attempts} attempts: {tries}')
    return complete


def select_by_toc(cut, protocol, complete):
    """(selected cut sections in document order, per-document record) from the model's choice."""
    docs = {}
    for s in cut:
        docs.setdefault(s['source_file'], []).append(s)
    keep, record = set(), []
    for document, doc_sections in docs.items():
        prompt = toc_prompt(protocol, document, doc_sections)
        text, meta = complete(prompt)
        try:
            reply = json.loads(text)
        except ValueError as exc:
            raise RuntimeError(f'table-of-contents selection for {document}: reply is not the requested JSON: {exc}')
        chosen = reply.get('selected') if isinstance(reply, dict) else None
        if not isinstance(chosen, list) or any(not isinstance(x, str) for x in chosen):
            raise RuntimeError(f'table-of-contents selection for {document}: selected must be a list of string identifiers')
        ids = {s['number']: s for s in doc_sections}
        unknown = [x for x in chosen if x not in ids]
        if unknown:
            raise RuntimeError(f'table-of-contents selection for {document}: unknown identifiers: {unknown}')
        known = list(dict.fromkeys(chosen))
        keep.update((document, x) for x in known)
        record.append(dict(document=document, headings=len(doc_sections), selected=known,
                           excluded=[x for x in ids if x not in known], unknown=[], prompt=prompt,
                           prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), reply=text, **meta))
    return [s for s in cut if (s['source_file'], s['number']) in keep], record


def build(doc_dir, out_path, select=None, protocol=None, complete=None, model=None):
    """Write (or reuse) the Stage B input from the documents in doc_dir.

    select=None keeps every section; select='toc_llm' keeps the sections a model picks from each
    document's table of contents (select_by_toc; `complete` and `protocol` required).
    The file carries its receipt under `_spec_sections`: the method, each input document's
    SHA-256 and the pdftotext version, and for a selection the model, the instructions' SHA-256
    and every reply. A file whose receipt matches the current documents (and selection) is
    reused as is, so a run on a host without pdftotext or the model can consume a cut made
    elsewhere. Returns (path, receipt, reused).
    """
    docs = input_documents(doc_dir)
    if not docs:
        raise ValueError(f'no input documents (*.pdf, *.txt, *.docx) in {doc_dir}')
    documents = [dict(file=d.name, sha256=_sha256(d)) for d in docs]
    if select not in (None, 'toc_llm'):
        raise ValueError(f'unknown selection {select!r}')
    wanted = dict(method=METHOD, documents=documents)
    if select:
        if not protocol:
            raise ValueError("select='toc_llm' needs `protocol`")
        wanted.update(selection_method=select, protocol=protocol, prompt_version=TOC_PROMPT_VERSION,
                      instructions_sha256=hashlib.sha256(TOC_SELECT_INSTRUCTIONS.encode()).hexdigest(), model=model)
    out_path = Path(out_path)
    if out_path.exists():
        existing = json.loads(out_path.read_text()).get('_spec_sections') or {}
        if all(existing.get(k) == v for k, v in wanted.items()) and (select or not existing.get('selection_method')):
            return out_path, existing, True
    cut = cut_documents(doc_dir)
    receipt = dict(wanted)
    if select:
        if complete is None or not protocol:
            raise ValueError("select='toc_llm' needs `complete` and `protocol`")
        chosen, record = select_by_toc(cut, protocol, complete)
        if not chosen:
            raise RuntimeError('table-of-contents selection returned no sections across the input documents')
        receipt.update(selection='sections chosen by the model from each table of contents; no GT',
                       headings=len(cut), per_document=record)
        cut = chosen
    else:
        receipt['selection'] = 'all sections; no model, no GT'
    sections = stage_b_sections(cut)
    receipt.update(pdftotext=_pdftotext_version() if any(d.suffix == '.pdf' for d in docs) else None,
                   sections=len(sections), chars=sum(len(s['content']) for s in sections))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pending = out_path.with_suffix('.pending')
    pending.write_text(json.dumps(dict(sections=sections, _spec_sections=receipt), ensure_ascii=False, indent=1) + '\n')
    pending.replace(out_path)
    return out_path, receipt, False
