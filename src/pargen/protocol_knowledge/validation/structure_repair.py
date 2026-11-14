"""Conservative repair of redundant ancestor child links; never infer a parent."""
import copy


def repair_redundant_ancestor_links(tree):
    candidate = copy.deepcopy(tree)
    nodes = candidate.get('nodes', [])
    if any(not isinstance(n, dict) or 'node_id' not in n for n in nodes):
        return candidate, []
    by_id = {n['node_id']: n for n in nodes}
    if len(by_id) != len(nodes):
        return candidate, []
    changes = []
    for parent in nodes:
        for child_id in list(parent.get('children_ids', [])):
            child = by_id.get(child_id)
            if child is None or child.get('parent_id') == parent['node_id']:
                continue
            # Only remove an extra ancestor edge when an independently
            # reciprocal, acyclic chain already connects child to ancestor.
            current = child
            visited = {child_id}
            while current.get('parent_id') in by_id:
                declared = by_id[current['parent_id']]
                if declared['node_id'] in visited or current['node_id'] not in declared.get('children_ids', []):
                    break
                visited.add(declared['node_id'])
                if declared['node_id'] == parent['node_id']:
                    parent['children_ids'].remove(child_id)
                    changes.append(dict(parent_id=parent['node_id'], child_id=child_id,
                                        retained_parent_id=child['parent_id'], reason='redundant_ancestor_link'))
                    break
                current = declared
    return candidate, changes
