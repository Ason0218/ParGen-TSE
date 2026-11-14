"""Protocol-neutral, bounded wire-layout checks; never edits generated code.

This is a static consistency gate, not proof of reachability or semantics.
Unsupported expressions/accessors remain explicit failures for fixed fields.
"""
from __future__ import annotations

import ast
import operator
import re

C_REPAIR_MAX_ROUNDS = 3
CHECKER_VERSION = 'fixed_layout_classified_feedback_v2'


def dynamic_layout(nodes, nid):
    """Only validated, bound runtime expressions are outside the fixed gate.

    Reuse the source audit's expression grammar. An unknown function/reference,
    cycle or invalid constant is NOT reclassified as dynamic to pass the gate.
    """
    from scripts.source_layout_consistency import symbolic_expression
    dynamic = False
    for prop in ('bit_start', 'size_bits'):
        if nodes[nid][prop] == 'variable' and prop == 'size_bits':
            dynamic = True
            continue
        try:
            number = resolve_property(nodes, nid, prop)
        except (ValueError, SyntaxError):
            parsed = symbolic_expression(nodes, nid, prop)
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'val'
                       for n in ast.walk(parsed)):
                raise ValueError('unsupported fixed expression, not a runtime-dependent layout')
            dynamic = True
        else:
            if number < 0 or prop == 'size_bits' and number == 0:
                raise ValueError('non-positive/negative span')
    return dynamic


def integer_expression(value):
    if type(value) is int:
        return value
    if not isinstance(value, str) or len(value) > 512:
        raise ValueError('not a bounded integer expression')
    tree = ast.parse(value.strip(), mode='eval')
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.LShift: operator.lshift,
           ast.RShift: operator.rshift, ast.BitOr: operator.or_, ast.BitAnd: operator.and_}
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            result = n.value
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            result = visit(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
        elif isinstance(n, ast.BinOp) and type(n.op) in ops:
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, (ast.LShift, ast.RShift)) and not 0 <= b <= 64:
                raise ValueError('unbounded shift')
            result = ops[type(n.op)](a, b)
        else:
            raise ValueError('unsupported expression')
        if abs(result) > 2 ** 64:
            raise ValueError('unbounded integer')
        return result
    try:
        return visit(tree.body)
    except (ZeroDivisionError, RecursionError) as exc:
        raise ValueError('invalid expression') from exc


def resolve_property(nodes, node_id, prop, trail=()):
    key = (node_id, prop)
    if key in trail or len(trail) > 100:
        raise ValueError('cyclic layout expression')
    value = nodes[node_id][prop]
    if isinstance(value, str):
        def replace(match):
            other = int(match[1] or match[4])
            attr = match[2] or match[3]
            return str(resolve_property(nodes, other, attr, (*trail, key)))
        value = re.sub(r'\b(\d+)\.(bit_start|size_bits)\b|\b(bit_start|size_bits)_(\d+)\b', replace, value)
    try:
        return integer_expression(value)
    except (SyntaxError, KeyError) as exc:
        raise ValueError('unsupported layout expression') from exc


def fixed_leaf_layouts(tree):
    nodes = {n['node_id']: n for n in tree['nodes']}
    rows, unresolved = [], []
    for node in nodes.values():
        if node.get('children_ids') or node.get('node_type') in {'protocol', 'header', 'variant', 'container'}:
            continue
        nid = node['node_id']
        try:
            if dynamic_layout(nodes, nid):
                unresolved.append(dict(node_id=nid, kind='dynamic', status='not_statically_verified',
                    reason='validated runtime-dependent layout; requires runtime evaluation',
                    bit_start=node.get('bit_start'), size_bits=node.get('size_bits')))
                continue
            start = resolve_property(nodes, nid, 'bit_start')
            width = resolve_property(nodes, nid, 'size_bits')
            if start < 0 or width <= 0:
                raise ValueError('non-positive/negative span')
            byte_count = (start % 8 + width + 7) // 8
            mask = (((1 << width) - 1) << (8 * byte_count - start % 8 - width)
                    if byte_count <= 8 else None)
            if start % 8 == 0 and width % 8 == 0:
                mask = 0
            if mask is None:
                raise ValueError('non-byte-aligned field larger than 64 bits')
            if node.get('byte_order', 'big') not in {'big', 'big-endian', 'network'}:
                raise ValueError('non-MSB0 byte order is not statically supported')
            rows.append(dict(node_id=nid, name=node['name'], bit_start=start, size_bits=width,
                             byte_offset=start // 8, byte_length=byte_count, mask=mask))
        except (ValueError, KeyError, SyntaxError) as exc:
            unresolved.append(dict(node_id=nid, kind='unsupported', reason=str(exc), bit_start=node.get('bit_start'),
                                   size_bits=node.get('size_bits')))
    return rows, unresolved


def layout_prompt(tree):
    import json
    rows, unresolved = fixed_leaf_layouts(tree)
    return ('\n<fixed_layout_contract>\n'
        'For every fixed leaf below, emit exactly one comment /* BS_LAYOUT node_id=N hf=hf_identifier */ '
        'binding its PKG ID to its real hf_register_info handle. Names are not used for matching. '
        'Use proto_tree_add_item(tree, hf_identifier, tvb, BYTE_OFFSET, BYTE_LENGTH, ENC_BIG_ENDIAN) '
        'with literal numeric offsets/lengths for these fixed absolute spans; retain conditional branch guards. '
        'Use the supplied MSB0 registration mask (0 means full bytes), never a low-bit placeholder. '
        'For scalar types use an FT_UINT width large enough for the whole accessed byte span; '
        'FT_BOOLEAN display width must equal the accessed bits and have a nonzero mask. '
        'Large byte-aligned payloads may use FT_BYTES. Preserve bounds checks. '
        'This contract does not license changing the source tree, ignoring variants, or adding unreachable reads. '
        'Dynamic fields still require parsing but are outside this fixed-span static check.\n' +
        'Only explicitly classified dynamic rows are excluded; unsupported/malformed expressions remain blocking. '
        'For FT_BOOLEAN the display member is a numeric bit width (e.g. 8), not BASE_HEX/BASE_DEC. '
        'A diagnostic about boolean display or accessor offset does not authorize changing a correct mask.\n' +
        json.dumps(dict(fixed=rows, dynamic_or_unresolved=unresolved), ensure_ascii=False) +
        '\n</fixed_layout_contract>\n')


_TOKEN = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
                    r'0[xX][0-9a-fA-F]+[uUlL]*|\d+[uUlL]*|[A-Za-z_]\w*|<<|>>|[^\s]', re.S)


def _parts(tokens, start, opener, closer):
    if tokens[start] != opener:
        raise ValueError('invalid C delimiter')
    depth, parts, current = [], [], []
    pairs = {'(': ')', '{': '}', '[': ']'}
    for index in range(start + 1, len(tokens)):
        token = tokens[index]
        if not depth and token == closer:
            return parts + [current], index
        if not depth and token == ',':
            parts.append(current)
            current = []
            continue
        if token in pairs:
            depth.append(pairs[token])
        elif token in pairs.values():
            if not depth or depth.pop() != token:
                raise ValueError('unbalanced C expression')
        current.append(token)
    raise ValueError('unterminated C expression')


def _number(tokens):
    return integer_expression(' '.join(re.sub(r'(?<=[0-9a-fA-F])[uUlL]+$', '', t)
                                       if re.match(r'^(?:0x|\d)', t, re.I) else t for t in tokens))


def check_c_layout(tree, files):
    rows, dynamic = fixed_leaf_layouts(tree)
    bindings, registers, calls, issues, diagnostics = {}, {}, {}, [], []
    for filename, source in files.items():
        if not filename.endswith('.c'):
            continue
        matches = list(_TOKEN.finditer(source))
        raw_tokens = [m.group() for m in matches]
        for token in raw_tokens:
            if not token.startswith('/*'):
                continue
            m = re.fullmatch(r'/\*\s*BS_LAYOUT\s+node_id=(\d+)\s+hf=(hf_\w+)\s*\*/', token)
            if m:
                bindings.setdefault(int(m[1]), []).append((filename, m[2]))
        tokens = [t for t in raw_tokens if not t.startswith(('/*', '//'))]
        lines = [source.count('\n', 0, m.start()) + 1 for m in matches
                 if not m.group().startswith(('/*', '//'))]
        for i, token in enumerate(tokens):
            # hf_register_info nested initializer, not comment/string claims.
            if token == '&' and i + 3 < len(tokens) and tokens[i+1].startswith('hf_') and tokens[i+2:i+4] == [',', '{']:
                try:
                    parts, _ = _parts(tokens, i+3, '{', '}')
                    if len(parts) >= 6 and len(parts[2]) == 1 and parts[2][0].startswith('FT_'):
                        registers.setdefault((filename, tokens[i+1]), []).append((parts, lines[i]))
                except ValueError:
                    issues.append(f'{filename}: malformed hf initializer')
            if token.startswith('proto_tree_add_') and i+1 < len(tokens) and tokens[i+1] == '(':
                try:
                    args, _ = _parts(tokens, i+1, '(', ')')
                    if len(args) > 1 and len(args[1]) == 1 and args[1][0].startswith('hf_'):
                        calls.setdefault((filename, args[1][0]), []).append((token, args, lines[i]))
                except ValueError:
                    issues.append(f'{filename}: malformed field accessor')
    checked = []
    for row in rows:
        nid, failures = row['node_id'], []
        def issue(code, prop, actual, expected, message, *, filename=None, line=None):
            failures.append(message)
            diagnostics.append(dict(code=code, node_id=nid, property=prop, actual=actual,
                                    expected=expected, file=filename, line=line, message=message))
        def number(tokens, prop, expected, filename, line):
            try:
                return _number(tokens)
            except (ValueError, SyntaxError):
                message = (f'{prop}: cannot statically resolve {" ".join(tokens)!r}; expected {expected}. '
                           'Repair this property only; preserve unrelated registration masks and branch guards.')
                issue('unresolved_constant', prop, ' '.join(tokens), expected, message, filename=filename, line=line)
                return None
        refs = bindings.get(nid, [])
        if len(refs) != 1:
            issue('binding_count', 'binding', len(refs), 1, 'missing or duplicate explicit node-to-hf binding')
        else:
            key = refs[0]
            if sum(key in value for value in bindings.values()) != 1:
                issue('shared_handle', 'binding', key[1], 'one PKG node', 'hf handle shared by multiple PKG nodes')
            regs = registers.get(key, [])
            access = calls.get(key, [])
            if len(regs) != 1 or not access:
                issue('registration_or_access_missing', 'binding', dict(registrations=len(regs), accessors=len(access)),
                      'one registration and at least one accessor', 'binding lacks unique registration or actual field accessor')
            else:
                reg, line = regs[0]
                filename = key[0]
                mask = number(reg[5], 'registration.mask', row['mask'], filename, line)
                if mask is not None and mask != row['mask']:
                    issue('mask_mismatch', 'registration.mask', mask, row['mask'],
                          f'registration mask != {hex(row["mask"])}', filename=filename, line=line)
                ft = reg[2][0]
                width = re.fullmatch(r'FT_(?:U?INT)(8|16|24|32|40|48|56|64)', ft)
                if width:
                    if int(width[1]) < row['byte_length'] * 8:
                        issue('scalar_width', 'registration.type', ft, row['byte_length'] * 8,
                              'registered scalar width smaller than accessed span', filename=filename, line=line)
                elif ft == 'FT_BOOLEAN':
                    display = number(reg[3], 'registration.boolean_display_bits', row['byte_length'] * 8, filename, line)
                    if not row['mask'] or display is not None and display != row['byte_length'] * 8:
                        issue('boolean_width', 'registration.boolean_display_bits', ' '.join(reg[3]),
                              row['byte_length'] * 8, 'boolean storage width/mask mismatch', filename=filename, line=line)
                elif ft not in {'FT_BYTES', 'FT_STRING'} or row['mask']:
                    issue('unsupported_type', 'registration.type', ft, 'compatible with audited fixed span',
                          'unsupported registration type for audited span', filename=filename, line=line)
                for fn, args, line in access:
                    if fn != 'proto_tree_add_item' or len(args) != 6:
                        issue('unsupported_accessor', 'accessor', fn, 'proto_tree_add_item with 6 arguments',
                              'unsupported accessor; fixed-span verification unresolved', filename=filename, line=line)
                        continue
                    for prop, tokens, expected in (('accessor.offset', args[3], row['byte_offset']),
                                                    ('accessor.length', args[4], row['byte_length'])):
                        actual = number(tokens, prop, expected, filename, line)
                        if actual is not None and actual != expected:
                            issue('span_mismatch', prop, actual, expected,
                                  f'actual accessor offset/length differs from tree: {prop}', filename=filename, line=line)
                    if args[5] not in (['ENC_BIG_ENDIAN'], ['ENC_NA']):
                        issue('encoding', 'accessor.encoding', ' '.join(args[5]), 'ENC_BIG_ENDIAN or ENC_NA',
                              'unsupported or inconsistent encoding', filename=filename, line=line)
        checked.append(dict(**row, status='failed' if failures else 'verified', errors=failures))
        issues.extend(f'node {nid}: {error}' for error in failures)
    # Unsupported fixed layouts must not silently become dynamic exclusions.
    for row in dynamic:
        if row['kind'] != 'dynamic':
            issues.append(f'node {row["node_id"]}: unsupported layout: {row["reason"]}')
            diagnostics.append(dict(code='unsupported_tree_layout', node_id=row['node_id'],
                property='tree.layout', actual=dict(bit_start=row['bit_start'], size_bits=row['size_bits']),
                expected='a supported fixed or validated runtime-dependent expression', message=row['reason']))
    return dict(ok=not issues and bool(rows), checker_version=CHECKER_VERSION, scope='fixed_leaf_registration_and_access_only',
                checked=checked, dynamic_or_unresolved=dynamic, issues=issues, diagnostics=diagnostics,
                coverage=dict(fixed=len(rows), fixed_verified=sum(r['status']=='verified' for r in checked),
                    dynamic_unverified=sum(r['kind']=='dynamic' for r in dynamic),
                    unsupported=sum(r['kind']!='dynamic' for r in dynamic)),
                runtime_layout_verified=False, branch_semantics_proven=False)


def repair_feedback(report):
    """Group common diagnostic properties without proposing protocol answers."""
    groups = {}
    for row in report['diagnostics']:
        key = (row['code'], row['property'])
        groups.setdefault(key, []).append(row)
    return dict(checker_version=CHECKER_VERSION, coverage=report['coverage'],
        groups=[dict(code=code, property=prop, affected_nodes=sorted({r['node_id'] for r in rows}),
                     evidence=rows) for (code, prop), rows in groups.items()],
        instructions='Fix the named property at its source location. Unresolved accessor expressions do not '
            'imply bad registration masks. Do not alter a correct mask or field width to repair an offset. '
            'Dynamic exclusions are unverified coverage, not obligations to replace lengths with constants.')


def repair_c_layout(tree, files, llm, state, out, profile=None, host_docs=''):
    """Bounded model-only repairs with checkpointable candidates and diagnostics."""
    import hashlib
    import json
    from .code_repair_agent import CodeRepairAgent, RepairDiagnostics
    from .source_sanity import generated_source_violations
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    origin = digest(dict(tree=tree, files=files))
    if state and state.get('origin_sha256') != origin:
        raise ValueError('Cannot reuse layout repairs from another tree/source')
    if not state:
        state.update(origin_sha256=origin, rounds=0, files=dict(files), attempts=[], checker_version=CHECKER_VERSION)
    if state.get('checker_version') != CHECKER_VERSION:
        raise ValueError('Explicit frozen checker revision required for legacy C layout state')
    if type(state.get('rounds')) is not int or not 0 <= state['rounds'] <= C_REPAIR_MAX_ROUNDS:
        raise ValueError('Invalid C layout repair budget')
    out.mkdir(parents=True, exist_ok=True)
    def record():
        (out / 'checkpoint.json').write_text(json.dumps(state, indent=2))
    while True:
        current = state['files']
        report = check_c_layout(tree, current)
        state['report'] = report
        record()
        if report['ok']:
            return dict(current)
        if not state.get('round_pending'):
            if state['rounds'] >= C_REPAIR_MAX_ROUNDS:
                raise RuntimeError('C fixed-layout repair budget exhausted')
            state['rounds'] += 1
            state['round_pending'] = True
            record()
        round_dir = out / f'round_{state["rounds"]}'
        round_dir.mkdir(exist_ok=True)
        (round_dir / 'before.json').write_text(json.dumps(current, indent=2))
        (round_dir / 'diagnostics.json').write_text(json.dumps(report, indent=2))
        diagnostics = RepairDiagnostics(tool_name='fixed_wire_layout', protocol_name='source_tree',
            errors=tuple(report['issues']), host_docs=host_docs, host_docs_max_chars=12000)
        changed = CodeRepairAgent(llm).repair_files(files=current, diagnostics=diagnostics,
            codegen_profile=profile, extra_rules=layout_prompt(tree) + '\n' + json.dumps(repair_feedback(report)) +
            '\nMake a local source patch justified by the tree, without changing protocol knowledge. '
            'Fix actual registrations and read expressions, not comments alone. '
            'Keep runtime branches, constraint checks, bounds checks and plugin entrypoints.', max_attempts=1, allow_new_files=False,
            prompt_path=round_dir / 'prompt.txt', response_path=round_dir / 'response.txt')
        state['round_pending'] = False
        candidate = dict(current, **(changed or {}))
        after = check_c_layout(tree, candidate)
        old_verified = sum(r['status'] == 'verified' for r in report['checked'])
        new_verified = sum(r['status'] == 'verified' for r in after['checked'])
        progress = (not generated_source_violations(candidate) and candidate != current and
                    (after['ok'] or new_verified > old_verified or len(after['issues']) < len(report['issues'])))
        state['attempts'].append(dict(round=state['rounds'], accepted=progress,
            candidate_sha256=digest(candidate), report=after))
        (round_dir / 'candidate.json').write_text(json.dumps(candidate, indent=2))
        if not progress:
            record()
            raise RuntimeError('C layout repair made no validated progress')
        state['files'] = candidate
        record()
