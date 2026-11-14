import logging
import json
import hashlib
import copy
import re
import os
from typing import Optional, List, Dict, Tuple, Set, Any

ISSUE_SCENARIO_TEXTS = {
    "parent_overflow": """Scenario: Parent container overflow / mis-sized header.
Validator errors: "Layout: Container_A(ID:10) children may exceed parent size"
Patch:
{
  "patch_metadata": { "description": "Repack header children and resize parent", "intent": "Fix overflow" },
  "node_updates": [
    { "node_id": 10, "property": "size_bits", "value": "48 + val(12)" },
    { "node_id": 20, "property": "bit_start", "value": "10.bit_start + 16" },
    { "node_id": 30, "property": "bit_start", "value": "10.bit_start + 16 + 20.size_bits" }
  ]
}""",
    "missing_length_binding": """Scenario: Missing length_of binding for variable payload.
Validator errors: "Semantics: Payload_B(ID:30) references val(12) but no length_of edge exists"
Patch:
{
  "patch_metadata": { "description": "Bind Payload_B to Length_Field", "intent": "Restore dependency" },
  "node_updates": [
    { "node_id": 30, "property": "size_bits", "value": "val(12)" }
  ],
  "new_edges": [
    { "src": 12, "dst": 30, "rel": "length_of", "formula": "val(12)", "message_type": "bidirectional" }
  ]
}""",
    "duplicate_root": """Scenario: Duplicate header/root nodes and mis-parented children.
Validator errors: "Structure: Multiple root nodes found" / "Child X has different parent"
Patch:
{
  "patch_metadata": { "description": "Merge duplicate header into canonical root", "intent": "Single root" },
  "node_updates": [
    { "node_id": 16, "property": "parent_id", "value": 1 },
    { "node_id": 1, "property": "children_ids", "value": [2,3,4,5,16] }
  ],
  "nodes_to_remove": [ { "node_id": 15 } ]
}""",
    "condition_overlap": """Scenario: Selector variants overlap / duplicate condition predicates.
Validator errors: "Layout: Variant_Request invalid overlap with sibling" / "condition_on edges reuse identical predicates"
Patch:
{
  "patch_metadata": { "description": "Route overlapping variants from one shared selector", "intent": "Mutually exclusive variants" },
  "node_updates": [
    { "node_id": 6, "property": "children_ids", "value": [7,40,41] },
    { "node_id": 40, "property": "parent_id", "value": 6 },
    { "node_id": 41, "property": "parent_id", "value": 6 }
  ],
  "nodes_to_remove": [ { "node_id": 70 }, { "node_id": 71 } ],
  "edge_removes": [
    { "src": 70, "dst": 40, "rel": "condition_on" },
    { "src": 71, "dst": 41, "rel": "condition_on" }
  ],
  "new_edges": [
    { "src": 7, "dst": 40, "rel": "condition_on", "formula": "val(7) == 1", "message_type": "response" },
    { "src": 7, "dst": 41, "rel": "condition_on", "formula": "val(7) == 129", "message_type": "response" }
  ]
}""",
    "source_backed_selector_fallback_overlap": """Scenario: A generic fallback/error variant overlaps a source-backed selector-specific variant.
Validator errors: "NON_EXCLUSIVE_VARIANT_CONDITIONS ... Non-exclusive conditions with Generic_Error_Response" or similar.

Correct repair pattern:
- Preserve source-backed packet-format variants.
- Narrow the fallback condition so it excludes every selector value handled by a specific sibling.
- Do not solve this by deleting the source-backed specific branch.

Patch:
{
  "patch_metadata": { "description": "Narrow the fallback selector guard", "intent": "Make fallback and specific variants mutually exclusive" },
  "edge_updates": [
    {
      "update_type": "modify",
      "edge_identifier": { "src": "<SEL_ID>", "dst": "<FALLBACK_VARIANT_ID>", "rel": "condition_on" },
      "new_properties": {
        "formula": "val(<SEL_ID>) >= <FALLBACK_LOW> and val(<SEL_ID>) != <SPECIFIC_ERROR_CODE>",
        "message_type": "response"
      }
    }
  ],
  "nodes_to_remove": [],
  "edge_removes": []
}""",
    "length_bound_fixed_payload": """Scenario: Length-bound payload modeled as a fixed-width field.
Validator errors: "Formula constraint cannot be satisfied" near a length/count field, while the evidence says "Length/Byte Count gives the byte length of Payload" and the payload has fixed size_bits.

Correct repair pattern:
- Keep the documented length/count relation.
- Make the payload/body field variable-sized or repeated according to the length/count scalar.
- Add or repair the length_of edge from the scalar length/count field to the payload/body target.
- Do not delete the length_relation/mod_alignment constraint to make validation pass.

Patch:
{
  "patch_metadata": { "description": "Bind variable payload size to its length/count field", "intent": "Make length relation executable" },
  "node_updates": [
    { "node_id": "<PAYLOAD_ID>", "property": "size_bits", "value": "val(<LEN_ID>) * 8" }
  ],
  "new_edges": [
    { "src": "<LEN_ID>", "dst": "<PAYLOAD_ID>", "rel": "length_of", "formula": "val(<LEN_ID>) * 8", "message_type": "<same message_type as payload>" }
  ],
  "edge_updates": [],
  "edge_removes": []
}""",
    "scalar_quantity_overlap": """Scenario: A small scalar quantity-like field overlaps with a variable-length sibling payload field in the same container.

Validator errors:
- "Structure: ... Invalid overlap with sibling ..."
- Both overlapping nodes share the same parent and message_type.
- One node is a fixed-width scalar (size_bits is a small constant like 8/16/32) that is used only inside size_bits / constraint expressions.
- The other node is variable-length and its size_bits formula references val(<SCALAR_NODE_ID>) (possibly multiplied by a constant factor).

Typical interpretations:
1. The scalar is a real on-wire header field which should appear before the payload.
2. The scalar is purely semantic / derived and should not occupy any on-wire bits (semantic-only field).
3. There are multiple quantity-like fields controlling the same payload length and one of them is redundant.

You must consider both possibilities (1) and (2) and choose a fix that:
- eliminates the physical overlap,
- keeps a single consistent length source for the payload,
- and passes both static and traffic validators.

Typical fixes:

A. Re-layout header vs payload (scalar is on-wire):
- Ensure the scalar quantity field starts where the parent’s header ends.
- Ensure the payload field starts immediately after the scalar:
  * payload.bit_start = scalar.bit_start + scalar.size_bits
- Ensure the parent container’s size_bits equals the sum of its children’s size_bits, instead of leaving gaps or overlaps.

B. Convert the scalar into a semantic-only quantity (no on-wire bits):
- Set the scalar’s size_bits to 0:
  * { "node_id": "<SCALAR_NODE_ID>", "property": "size_bits", "value": "0" }
- Remove the scalar from the parent’s layout / size_bits formula:
  * Do not add scalar.size_bits when computing the parent size.
- Keep uses of val(<SCALAR_NODE_ID>) in size_bits or constraint formulas if they still make semantic sense.

C. Prefer intra-branch scalar length sources over cross-branch ones:
- If a variable-length payload currently uses a quantity field from a different branch or message direction, consider:
  * switching to a scalar sibling in the same branch; or
  * converting the cross-branch quantity field into a semantic-only field (size_bits = 0) so it does not interfere with layout.

Patch skeleton examples:

1) Semantic-only quantity:
{
  "patch_metadata": {
    "description": "Treat scalar quantity as semantic-only to resolve overlap with payload",
    "intent": "Remove physical overlap while keeping a single length source"
  },
  "new_nodes": [],
  "node_updates": [
    { "node_id": "<SCALAR_NODE_ID>", "property": "size_bits", "value": "0" },
    { "node_id": "<PARENT_ID>", "property": "size_bits", "value": "<EXPRESSION_SUM_OF_CHILDREN>" }
  ],
  "new_edges": [],
  "edge_updates": [],
  "edge_removes": []
}

2) Re-layout header + payload:
{
  "patch_metadata": {
    "description": "Re-layout scalar quantity and payload to remove overlap",
    "intent": "Header-then-payload, contiguous"
  },
  "new_nodes": [],
  "node_updates": [
    { "node_id": "<SCALAR_NODE_ID>", "property": "bit_start", "value": "<PARENT_ID>.bit_start + <HEADER_PREFIX_BITS>" },
    { "node_id": "<PAYLOAD_NODE_ID>", "property": "bit_start", "value": "<SCALAR_NODE_ID>.bit_start + <SCALAR_NODE_ID>.size_bits" },
    { "node_id": "<PARENT_ID>", "property": "size_bits", "value": "<SUM_OF_CHILD_SIZE_BITS>" }
  ],
  "new_edges": [],
  "edge_updates": [],
  "edge_removes": []
}
""",
    "length_of_src_mismatch": """Scenario: length_of edge references wrong source node.
Validator errors: "Graph: Payload(ID:30): length_of formula references [14] but src is 20"
Patch:
{
  "patch_metadata": { "description": "Correct length_of source", "intent": "Align edge src with formula" },
  "edge_removes": [
    { "src": 20, "dst": 30, "rel": "length_of" }
  ],
  "new_edges": [
    { "src": 14, "dst": 30, "rel": "length_of", "formula": "val(14)", "message_type": "bidirectional" }
  ]
}""",
    "length_of_duplicates": """Scenario: Multiple length_of edges targeting same node.
Validator errors: "Graph: Payload(ID:30): Multiple length_of bindings to node 30"
Patch:
{
  "patch_metadata": { "description": "Consolidate length_of bindings", "intent": "Single sizing source" },
  "edge_removes": [
    { "src": 11, "dst": 30, "rel": "length_of" }
  ],
  "edge_updates": [
    { "update_type": "modify", "edge_identifier": { "src": 12, "dst": 30, "rel": "length_of" }, "new_properties": { "formula": "val(12) * 8" } }
  ]
}""",
    "traffic_parsing_blocked": """Scenario: Dynamic parsing blocked because a field size or type is incorrect.
Validator errors: "Semantics: Parsing blocked at Node X: Stream ended unexpectedly"
Patch:
{
  "patch_metadata": { "description": "Fix size of Node X or previous field", "intent": "Fix parsing block" },
  "node_updates": [
    { "node_id": "<Node_X_ID>", "property": "size_bits", "value": "val(<Length_Field_ID>) * 8" }
  ]
}""",
    "traffic_constraint_failed": """Scenario: Dynamic value violates constraint (likely endianness or strict range).
Validator errors: "Semantics: Constraint failed for Node Y: value 256 violated value < 255"
Patch:
{
  "patch_metadata": { "description": "Correct endianness for Node Y", "intent": "Fix constraint violation" },
  "node_updates": [
    { "node_id": "<Node_Y_ID>", "property": "byte_order", "value": "little" }
  ]
}""",
    "traffic_coverage_gap": """Scenario: Bytes remain unparsed at the end of the packet.
Validator errors: "Coverage: Parsing finished but 4 bytes remain unparsed"
Patch:
{
  "patch_metadata": { "description": "Add missing Checksum/Footer field", "intent": "Complete coverage" },
  "new_nodes": [
    { "node_id": 900, "name": "Checksum", "node_type": "field", "data_type": "uint32", "bit_start": "...", "size_bits": 32, "parent_id": "..." }
  ]
}""",
    "variant_body_double_selector_skip": """Scenario: A selector-controlled variant body skips the selector twice.
Validator errors include "VARIANT_BODY_DOUBLE_SELECTOR_SKIP" or "variant body appears to skip the controlling selector twice".

What this means:
- The selector/tag/opcode field is modeled once in the common parent.
- The variant already starts at selector_end.
- The first real child inside the variant starts at variant_start + selector_size, leaving an unmodeled leading gap.

Correct repair pattern:
- Do not duplicate the selector inside every variant.
- Shift the variant's child subtree earlier by selector_size bits, or insert a documented common field at the beginning of the variant body if the gap is real.

Patch shape:
{
  "patch_metadata": { "description": "Remove double selector offset from variant body", "intent": "Fix selector-controlled variant layout" },
  "node_updates": [
    { "node_id": "<first_child_id>", "property": "bit_start", "value": "<variant_id>.bit_start + 0" },
    { "node_id": "<next_child_id>", "property": "bit_start", "value": "<variant_id>.bit_start + <previous_child_size>" }
  ]
}""",
    "non_executable_constraint_text": """Scenario: A `constraints` entry contains prose instead of an executable constraint.
Validator errors include "NON_EXECUTABLE_CONSTRAINT_TEXT" or "descriptive prose, not an executable constraint".

What this means:
- The `constraints` array is reserved for machine-checkable constraints.
- Prose such as `description: ... calculated as ...` cannot be evaluated by the traffic interpreter.
- Pure explanations should be moved to the node `description`; real constraints should become `enum:`, `range:`, or `formula:` strings using node IDs.

Correct repair pattern:
- If the prose is just explanatory, remove it from `constraints` and append/preserve it in `description`.
- If it states a relation, convert it to an executable form, e.g. `formula: val(<ByteCount_ID>) == ((val(<Quantity_ID>) + 7) / 8)` or a size/length_of binding.
- Do not leave dictionaries or natural-language strings inside `constraints` unless they are normalized enum/range dictionaries supported by the schema.

Patch shape:
{
  "patch_metadata": { "description": "Normalize prose constraint into executable schema", "intent": "Fix non-executable constraint" },
  "node_updates": [
    { "node_id": "<NODE_ID>", "property": "constraints", "value": ["<existing executable constraints>", "formula: <executable relation using val(ID)>"] },
    { "node_id": "<NODE_ID>", "property": "description", "value": "<existing description plus any non-constraint prose>" }
  ]
}""",
    "traffic_length_mismatch": """Scenario: Traffic parsing shows a node's size_bits expression reads past the end of the packet (not enough bits remain).

Validator errors typically look like:
- "[TRAFFIC][STRUCTURE][ERROR] node=<ID> size_bits expression leads to read past end of packet in sample #<N> ..."
- Messages containing "read past end of packet", "not enough bits", or "size_bits expression leads to out-of-bounds".

What this usually means:
- The node is variable-length and its size_bits formula does not match real packets, or
- The scalar quantity field that drives the length is mis-modeled (bit_start/size_bits wrong), or
- The parent container size leaves gaps or double-counts bytes.

When you see this, do NOT just loosen constraints or inflate packet length. Instead:

1) Inspect the failing node's size_bits expression:
- Does it reference a scalar sibling like val(<QUANTITY_NODE_ID>)?
- Does bits_needed exceed the remaining bits between cursor and packet end?

2) Inspect scalar quantity-like fields used in that expression:
- Are they small fixed-width scalars (8/16/32 bits) placed before the payload?
- Do their traffic values match the payload length?
- If they are purely semantic, consider making them semantic-only (size_bits=0).

3) Candidate fixes:

A. Fix the size_bits formula:
- Use the correct quantity/byte-count field in the same branch.
- Adjust multipliers to match traffic (e.g., *8 vs *16).
- Ensure size_bits never exceeds remaining bits.

B. Fix or convert the quantity field:
- If on-wire: correct its size_bits/bit_start so it precedes the payload.
- If semantic-only: set size_bits=0 so it doesn't consume bits.

C. Adjust the parent container layout:
- Make children contiguous: payload.bit_start = previous_sibling.bit_start + previous_sibling.size_bits.
- Set parent.size_bits to the sum of its children when they are present.

Patch skeleton example:
{
  "patch_metadata": {
    "description": "Align size_bits and quantity field with traffic to avoid out-of-bounds read",
    "intent": "Match node length to actual packets"
  },
  "new_nodes": [],
  "node_updates": [
    { "node_id": "<PAYLOAD_NODE_ID>", "property": "size_bits", "value": "<CORRECT_EXPRESSION>" },
    { "node_id": "<QUANTITY_NODE_ID>", "property": "size_bits", "value": "0" },  # optional semantic-only
    { "node_id": "<PARENT_ID>", "property": "size_bits", "value": "<SUM_OF_CHILDREN_SIZE_BITS>" }
  ],
  "new_edges": [],
  "edge_updates": [],
  "edge_removes": []
}
""",
    "doc_constraint_coverage": """Scenario: Documentation constraint coverage is missing a guarded field constraint.
Validator errors: "Documentation constraint appears uncovered: [Some Message] Code in {0|1} under Type in {11} ..."

Correct repair pattern:
- Prefer the SAFE REPAIR text in the issue. It lists exact variant IDs, target field IDs, and formula strings.
- Respect selector scope exactly: a constraint documented for selector value K must only be added to the variant whose condition_on guard includes K.
- Do not attach a fixed-header/global field constraint to a TLV/option variant just because that variant has a tag/kind selector.
- If the target field is shared in a header, do NOT widen/delete that shared field's constraints to a union/range.
- Append a target-specific `formula:` to the selected variant's constraints:
  * GOOD: `formula: val(<Code_ID>) == 0 or val(<Code_ID>) == 1`
  * GOOD: `formula: val(<Checksum_ID>) >= 0 and val(<Checksum_ID>) <= 65535`
  * BAD: adding `enum: 0|1` to the variant node (not target-specific)
  * BAD: adding `condition_on` edges from selector to leaf fields.
- Preserve existing constraints. If a shared constraint must be moved, replace it with equivalent scoped formulas on the branches where it still applies.

Patch skeleton:
{
  "patch_metadata": {
    "description": "Add scoped formula for documented guarded constraint",
    "intent": "Doc constraint coverage"
  },
  "node_updates": [
    {
      "node_id": "<VARIANT_ID_FROM_SAFE_REPAIR>",
      "property": "constraints",
      "value": ["<existing variant constraints>", "formula: val(<TARGET_FIELD_ID>) == <VALUE> or val(<TARGET_FIELD_ID>) == <VALUE2>"],
      "reason": "Covers documented guarded field constraint without widening the shared header field"
    }
  ],
  "new_nodes": [],
  "new_edges": [],
  "edge_updates": [],
  "edge_removes": [],
  "nodes_to_remove": []
}""",
    "doc_relational_obligation": """Scenario: Documentation prose states a relation between fields, but the tree does not encode it.
Validator errors: "Documentation relational obligation appears uncovered: ... Source relation: 'If Flag A is set, Flag B must be zero.'"

Correct repair pattern:
- Prefer the RELATIONAL REPAIR metadata in the issue. It lists field_node_ids and the quoted source relation.
- Add an explicit scoped `constraint_ir:` on the smallest correct variant/container. Add an executable `formula:` as well only when it is safe with the shown node IDs. Do not make a consequent unconditional just because it mentions a shared field.
- `val(ID)` may reference only scalar parsed-value nodes: selector/code/type/length/checksum/fixed scalar field. Do not write `val(ID)` over payload/body/container/data/variant nodes; bind to the selector/length/scalar child or use `constraint_ir:` only.
- For conditional rules, encode implication with the documented trigger:
  * GOOD: `constraint_ir: {{"kind":"conditional_zero","expr":"val(<FLAG_A_ID>) == 1 => val(<FLAG_B_ID>) == 0","field_ids":[<FLAG_A_ID>,<FLAG_B_ID>]}}`
  * GOOD: `formula: val(<FLAG_A_ID>) != 1 or val(<FLAG_B_ID>) == 0`
- For mutual exclusion, encode one relation over the two fields:
  * GOOD: `constraint_ir: {{"kind":"mutual_exclusion","expr":"not (val(<POLL_ID>) == 1 and val(<FINAL_ID>) == 1)","field_ids":[<POLL_ID>,<FINAL_ID>]}}`
  * GOOD: `formula: not (val(<POLL_ID>) == 1 and val(<FINAL_ID>) == 1)`
  * GOOD: `formula: val(<POLL_ID>) + val(<FINAL_ID>) <= 1`
- For length/count relations, bind the value or size expression to documented fields:
  * GOOD: `constraint_ir: {{"kind":"length_relation","expr":"val(<LEN_ID>) == 24 + val(<AUTH_LEN_ID>)","field_ids":[<LEN_ID>,<AUTH_LEN_ID>]}}`
  * GOOD: `formula: val(<LEN_ID>) == 24 + val(<AUTH_LEN_ID>)`
  * GOOD: node `size_bits = (val(<LEN_ID>) - <HEADER_BYTES>) * 8` when the relation is about payload size.
- For alignment/padding rules, use modulo/zero constraints on the modeled length/offset/padding bytes.
- Preserve existing constraints and children. If a relation field is absent, add the missing on-wire field or scoped variant first; do not invent protocol-specific hidden constants not present in the quoted evidence.

Patch skeleton:
{
  "patch_metadata": {
    "description": "Encode documented relational obligation",
    "intent": "Doc relational obligation"
  },
  "node_updates": [
    {
      "node_id": "<SMALLEST_SCOPE_NODE_ID>",
      "property": "constraints",
      "value": ["<existing constraints>", "constraint_ir: {\"kind\":\"<RELATION_KIND>\",\"expr\":\"<RELATION_OVER_FIELD_NODE_IDS>\",\"field_ids\":[<IDS>]}"],
      "reason": "Covers documented prose relation without hard-coded protocol logic"
    }
  ],
  "new_nodes": [],
  "new_edges": [],
  "edge_updates": [],
  "edge_removes": [],
  "nodes_to_remove": []
}""",
    "doc_structure_alignment": """Scenario: Documentation lists an on-wire field or guarded body variant that is missing or weakly named in the tree.
Validator errors:
- "Documentation structure appears uncovered: [Some Message] variant ... does not contain documented field ..."
- "Documentation structure has weak field-name alignment: ... appears to be Existing_Node(ID:...) ..."

Correct repair pattern:
- Prefer the SAFE STRUCTURAL REPAIR text in the issue. It lists the target variant, canonical field name, size_bits, and any documented enum/range constraint.
- Treat each packet-format row as its own guarded alternative unless the rows clearly share a generic TLV/length/data envelope. For TLV-style tables, first reuse an existing generic/catch-all branch such as `Normal_TLV`, `Other_Option`, or `TLV_Option` if it already has selector + length + data fields.
- For an existing TLV catch-all, prefer changing its existing `condition_on` formula so it covers the documented length-bearing selector values. Preserve the existing Length/Data nodes and their current size_bits semantics unless the documentation explicitly says the length includes/excludes the header.
- Do not add a parallel `TLV_Option`/`Other_Option` branch when an existing generic TLV branch can be repaired by an `edge_updates` formula change.
- In a TLV catch-all, make the condition cover all length-bearing values for the focused issue. Do not write a catch-all that only excludes the specific rows you just modeled (for example `type != 130 and type != 136`); exclude documented single-byte/no-body values and unrelated exact variants as needed.
- If the issue says `repair_phase=selector_shell` or "selector_shell phase", do only the selector shell step: add/rename zero-size variant containers and `condition_on` edges for the documented selector values. Do not add any body fields in that patch; body fields belong to a later `body_expansion` issue.
- In selector-shell phase, do not create or edit Length/Data/TLV/Option_Item body nodes. A selector shell is only a zero-size selected container plus its `condition_on` edge; it must preserve the existing generic body/TLV path unchanged.
- Selector shell variants should be siblings under the selector's enclosing container, not children of the selector leaf. If the selector is inside a fixed/common header, put the shell under the header's parent and start it at the header end. Otherwise use `bit_start = <selector_id>.bit_start + <selector_id>.size_bits`, `size_bits = 0`, and preserve the existing generic TLV/body path until the body-expansion phase.
- If the issue says `repair_phase=body_expansion`, refine the existing selected variant in place and keep its `condition_on` edge. Do not create a second variant for the same selector value.
- When adding or refining a TLV/body container with a Length field and payload children, bind the container/body span in the same patch before adding typed children: use the existing length/count relation mechanism (`length_of`, `size_bits: val(<LEN_ID>) * 8`, or an equivalent `constraint_ir` over the Length and Data nodes). If that binding cannot be expressed safely from the documented issue, stop at the zero-size selector shell and leave body expansion for a later issue.
- Do not add fields from one selector value's row to sibling variants for other selector values.
- If the issue says weak name alignment, rename or alias the existing node. Preserve its node_id, size_bits, children, and constraints; do not duplicate the same wire bytes.
- If the issue says missing scoped field, add the field under the listed variant in the documented order. A variant-level `formula:` is not a substitute for a real on-wire child field.
- If the issue says missing guarded variant, add a variant selected by the documented selector value and include the listed body fields.
- If the issue says repeated_section_shell or missing repeated format, add a distinct repeated section/container for the documented entry format, add the listed body fields in order, and connect the documented count field to that container with `repeat_count` or `constraint_ir` count relation. Do not reuse another record format that has extra fields.
- For missing guarded variants, keep the selector as a leaf field. Put the variant under the selector's enclosing container, not under the selector leaf, and set the variant/container `size_bits` to a concrete sum of its documented children whenever possible.
- The variant body's first field starts after the selector field unless the documentation explicitly shows another leading field.

Patch skeleton:
{
  "patch_metadata": {
    "description": "Align guarded body structure with documented packet format",
    "intent": "Doc structure alignment"
  },
  "node_updates": [
    {
      "node_id": "<EXISTING_WEAK_NODE_ID>",
      "property": "name",
      "value": "<CANONICAL_NAME_FROM_SAFE_REPAIR>",
      "reason": "Align existing wire field name with documentation"
    }
  ],
  "new_nodes": [
    {
      "node_id": "<NEW_ID>",
      "name": "<CANONICAL_NAME_FROM_SAFE_REPAIR>",
      "node_type": "field",
      "parent_id": "<VARIANT_ID_FROM_SAFE_REPAIR>",
      "children_ids": [],
      "size_bits": "<SIZE_BITS_FROM_SAFE_REPAIR>",
      "data_type": "uint",
      "constraints": ["<DIRECT_CONSTRAINT_IF_PRESENT>"],
      "message_type": "bidirectional"
    }
  ],
  "new_edges": [],
  "edge_updates": [],
  "edge_removes": [],
  "nodes_to_remove": []
}""",
    "explicit_dependency_missing": """Scenario: Formula references a node value (val(X)) but the corresponding graph edge is missing.
Validator errors: "Semantics: Node_Y(ID:<Y>): Size expression references val(<X>) but no length_of edge exists"
Patch:
{
  "patch_metadata": { "description": "Add missing length_of edge to support size formula", "intent": "Fix dependency" },
  "new_edges": [
    { "src": "<X>", "dst": "<Y>", "rel": "length_of", "formula": "val(<X>)", "message_type": "bidirectional" }
  ]
}""",
    "header_overflow_reparent": """Scenario: Fixed-size Header contains a variable or extra child that causes overflow.
Validator errors: "Layout: Header(ID:<H>): Children may exceed parent size... Offending children: Field_C(ID:<C>...)"
Patch:
{
  "patch_metadata": { "description": "Move Field_C out of fixed Header into PDU Payload", "intent": "Fix container overflow" },
  "node_updates": [
    { "node_id": "<C>", "property": "parent_id", "value": "<PDU_ID>" },
    { "node_id": "<H>", "property": "children_ids", "value": ["...remove <C>..."] },
    { "node_id": "<PDU_ID>", "property": "children_ids", "value": ["...add <C>..."] }
  ]
}""",
    "sync_size_with_edge": """Scenario: Field has 'variable' size but a valid length_of edge exists.
Validator errors: "Semantics: Field_A(ID:<A>) has size_bits=\"variable\" but also a length_of edge... Please move the length formula"
Patch:
{
  "patch_metadata": { "description": "Replace 'variable' in size_bits with the explicit formula from the length_of edge", "intent": "Sync size definition" },
  "node_updates": [
    { "node_id": "<A>", "property": "size_bits", "value": "val(<Len_ID>) * 8" }
  ]
}""",
    "relax_selector_constraint": """Scenario: Variant condition conflicts with parent selector's strict constraints.
Validator errors: "Semantics: Variant_B(ID:<B>): Variant condition ... cannot be satisfied within selector constraints"
Patch:
{
  "patch_metadata": { "description": "Relax selector constraints to include the new variant value", "intent": "Fix unsatisfiable condition" },
  "node_updates": [
    { "node_id": "<Selector_ID>", "property": "constraints", "value": ["enum: <existing_values>|<new_needed_value>"] }
  ]
}""",
    "self_reference_error": """Scenario: Agent incorrectly linked a node to itself.
Validator errors: "Edge 11->11 (length_of) is self-referential"
Patch:
{
  "patch_metadata": { "description": "Remove invalid self-referential edge", "intent": "Fix graph cycle" },
  "edge_removes": [
    { "src": 11, "dst": 11, "rel": "length_of" }
  ],
  "node_updates": [
    { "node_id": 11, "property": "size_bits", "value": "val(<ACTUAL_LENGTH_SOURCE_ID>) * 8" }
  ]
}""",
    "multiple_length_bindings": """Scenario: Multiple edges define size for the same node.
Validator errors: "Graph: Payload(ID:30): Multiple length_of bindings to node 30"
Patch:
{
  "patch_metadata": { "description": "Remove stale length binding, keep only the correct one", "intent": "Resolve conflict" },
  "edge_removes": [
    { "src": <WRONG_SRC_ID>, "dst": 30, "rel": "length_of" }
  ]
}""",
    "propagate_parent_length": """Scenario: Child has size but Parent container size is unknown.
Validator errors: "Layout: Child(ID:20) may extend beyond parent Container(ID:10) (parent length unknown)"
Patch:
{
  "patch_metadata": { "description": "Propagate length formula from PDU to specific Variant container", "intent": "Define parent size" },
  "node_updates": [
    { "node_id": 10, "property": "size_bits", "value": "val(<LENGTH_FIELD_ID>) * 8 - <HEADER_SIZE>" }
  ],
  "new_edges": [
    { "src": <LENGTH_FIELD_ID>, "dst": 10, "rel": "length_of", "formula": "val(<src>) * 8 - <HEADER_SIZE>", "message_type": "bidirectional" }
  ]
}""",
    "simplify_pdu_size": """Scenario: PDU/Container size formula is overly complex (using max/sum of many children) causing missing dependency errors.
Validator errors: "Semantics: PDU(ID:<ID>): Size expression references val(...) but no length_of edge exists"
Patch:
{
  "patch_metadata": { "description": "Simplify PDU size to rely on Header Length field instead of summing children", "intent": "Simplify structure" },
  "node_updates": [
    { "node_id": "<ID>", "property": "size_bits", "value": "(val(<LENGTH_FIELD_ID>) - <OFFSET>) * 8" }
  ],
  "edge_removes": [
    { "src": "<OLD_SRC_ID>", "dst": "<ID>", "rel": "length_of" }
  ],
  "new_edges": [
    { "src": "<LENGTH_FIELD_ID>", "dst": "<ID>", "rel": "length_of", "formula": "(val(<src>) - <OFFSET>) * 8", "message_type": "bidirectional" }
  ]
}""",
}

ISSUE_SCENARIO_RULES = [
    {"scenario": "doc_relational_obligation", "pattern": r"documentation relational obligation appears uncovered"},
    {"scenario": "doc_relational_obligation", "pattern": r"doc_relational_obligation"},
    {"scenario": "doc_structure_alignment", "pattern": r"documentation structure appears uncovered"},
    {"scenario": "doc_structure_alignment", "pattern": r"documentation structure has weak field-name alignment"},
    {"scenario": "doc_structure_alignment", "pattern": r"doc_structure_alignment"},
    {"scenario": "doc_constraint_coverage", "pattern": r"documentation constraint appears uncovered"},
    {"scenario": "doc_constraint_coverage", "pattern": r"doc_constraint_coverage"},
    {"scenario": "parent_overflow", "pattern": r"children may exceed parent size"},
    {"scenario": "missing_length_binding", "pattern": r"no length_of edge exists"},
    {"scenario": "duplicate_root", "pattern": r"multiple root nodes found"},
    {"scenario": "duplicate_root", "pattern": r"child \d+ has different parent"},
    {"scenario": "source_backed_selector_fallback_overlap", "pattern": r"non_exclusive_variant_conditions"},
    {"scenario": "source_backed_selector_fallback_overlap", "pattern": r"non-exclusive conditions with "},
    {"scenario": "length_bound_fixed_payload", "pattern": r"formula constraint cannot be satisfied"},
    {"scenario": "length_bound_fixed_payload", "pattern": r"byte count gives the byte length"},
    {"scenario": "condition_overlap", "pattern": r"condition_on"},
    {"scenario": "scalar_quantity_overlap", "pattern": r"invalid overlap with sibling"},
    {"scenario": "length_of_src_mismatch", "pattern": r"length_of formula references"},
    {"scenario": "length_of_duplicates", "pattern": r"multiple length_of bindings"},
    {"scenario": "traffic_parsing_blocked", "pattern": r"parsing blocked"},
    {"scenario": "traffic_parsing_blocked", "pattern": r"stream ended unexpectedly"},
    {"scenario": "traffic_constraint_failed", "pattern": r"constraint failed"},
    {"scenario": "traffic_coverage_gap", "pattern": r"remain[s]? unparsed"},
    {"scenario": "variant_body_double_selector_skip", "pattern": r"VARIANT_BODY_DOUBLE_SELECTOR_SKIP"},
    {"scenario": "variant_body_double_selector_skip", "pattern": r"variant body appears to skip the controlling selector twice"},
    {"scenario": "non_executable_constraint_text", "pattern": r"NON_EXECUTABLE_CONSTRAINT_TEXT"},
    {"scenario": "non_executable_constraint_text", "pattern": r"descriptive prose, not an executable constraint"},
    {"scenario": "non_executable_constraint_text", "pattern": r"prose/dict metadata, not an executable constraint"},
    {"scenario": "explicit_dependency_missing", "pattern": r"references val\(.*\) but no length_of edge exists"},
    {"scenario": "explicit_dependency_missing", "pattern": r"preserve the length_of relation"},
    {"scenario": "header_overflow_reparent", "pattern": r"children may exceed parent size"},
    {"scenario": "header_overflow_reparent", "pattern": r"offending children"},
    {"scenario": "sync_size_with_edge", "pattern": r"move the length formula into size_bits"},
    {"scenario": "relax_selector_constraint", "pattern": r"cannot be satisfied within selector constraints"},
    {"scenario": "self_reference_error", "pattern": r"is self-referential"},
    {"scenario": "multiple_length_bindings", "pattern": r"Multiple length_of bindings"},
    {"scenario": "propagate_parent_length", "pattern": r"parent length unknown"},
    {"scenario": "simplify_pdu_size", "pattern": r"Size expression .* references val.* but no length_of edge exists"},
    {"scenario": "traffic_length_mismatch", "pattern": r"read past end of packet"},
    {"scenario": "traffic_length_mismatch", "pattern": r"size_bits expression leads to out-of-bounds"},
    {"scenario": "traffic_length_mismatch", "pattern": r"not enough bits"},
]


def _build_node_maps(tree: dict) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """Return node lookup and depth map (root depth=0)."""
    nodes = tree.get("nodes", []) if isinstance(tree, dict) else []
    node_map: Dict[str, dict] = {}
    parent_map: Dict[str, Optional[str]] = {}
    for n in nodes:
        nid = n.get("node_id")
        if nid is None:
            continue
        nid_str = str(nid)
        node_map[nid_str] = n
        parent_map[nid_str] = str(n.get("parent_id")) if n.get("parent_id") is not None else None

    depth_cache: Dict[str, int] = {}

    def _depth(nid: str, visited: Optional[Set[str]] = None) -> int:
        if nid in depth_cache:
            return depth_cache[nid]
        if visited is None:
            visited = set()
        if nid in visited:
            # Cycle detected; return safe fallback depth
            return 0
        visited.add(nid)
        parent = parent_map.get(nid)
        if parent is None or parent not in parent_map:
            depth_cache[nid] = 0
        else:
            depth_cache[nid] = _depth(parent, visited) + 1
        return depth_cache[nid]

    for nid in parent_map:
        _depth(nid)

    return node_map, depth_cache


def _issue_priority(issue: dict, node_map: Dict[str, dict], depth_map: Dict[str, int]) -> Tuple[int, int, int, int]:
    """Priority tuple: lower is higher priority (depth -> bit_start -> -hits -> severity)."""
    severity_order = {"ERROR": 0, "WARN": 1, "HINT": 2}
    issue_type_order = {"STRUCTURE": 0, "SEMANTICS": 1, "COVERAGE": 2, "WARNING": 3}

    target = issue.get("target")
    if not isinstance(target, dict):
        target = {}
    node_id = target.get("identifier")
    node_id_str = str(node_id) if node_id is not None else None
    depth = depth_map.get(node_id_str, 10**6) if node_id_str else 10**6

    bit_start_val = 10**6
    if node_id_str and node_id_str in node_map:
        bs = node_map[node_id_str].get("bit_start")
        try:
            bit_start_val = int(bs)
        except Exception:
            bit_start_val = 10**6

    # Extract total_hits from description if present
    import re as _re
    hits = 0
    desc = issue.get("description")
    if isinstance(desc, str):
        m = _re.search(r"total_hits=(\d+)", desc)
        if m:
            try:
                hits = int(m.group(1))
            except Exception:
                hits = 0

    severity = severity_order.get(str(issue.get("severity") or "").upper(), 3)
    itype = issue_type_order.get(str(issue.get("type") or "").upper(), 4)

    return (depth, bit_start_val, -hits, severity, itype)

def build_issue_example_block(issues_payload, validator_errors, validator_extras):
    texts: List[str] = []
    for issue in issues_payload or []:
        if isinstance(issue, dict):
            description = issue.get("description")
            if isinstance(description, str):
                texts.append(description.lower())
    for err in validator_errors or []:
        if isinstance(err, str):
            texts.append(err.lower())
    for extra in validator_extras or []:
        if isinstance(extra, str):
            texts.append(extra.lower())

    matched: List[str] = []
    seen = set()
    for text in texts:
        for rule in ISSUE_SCENARIO_RULES:
            scenario = rule["scenario"]
            if scenario in seen:
                continue
            if re.search(rule["pattern"], text, re.IGNORECASE):
                snippet = ISSUE_SCENARIO_TEXTS.get(scenario)
                if snippet:
                    matched.append(snippet)
                    seen.add(scenario)
    if not matched:
        return ""
    return "### ISSUE-SPECIFIC FEWSHOT EXAMPLES\n" + "\n\n".join(matched) + "\n"


_PATCH_SECTION_TEXT_KEYS = {
    "content",
    "combined_text",
    "raw_content",
    "text",
    "raw_text",
    "summary",
    "raw_summary",
}
_PATCH_REPAIR_LINE_RE = re.compile(
    r"\b("
    r"type|code|kind|tag|opcode|op\s*code|function(?:\s+code)?|"
    r"length|size|count|must|shall|required|zero|range|valid|"
    r"bits?|bytes?|format|field|option|variant"
    r")\b|[+\-|]{3,}|^\s*[01xX]{4,}\s*$|=\s*(?:0x[0-9a-fA-F]+|\d+)",
    re.IGNORECASE,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip() or str(default))
    except Exception:
        return default


def _compact_repair_text(text: Any, *, limit: int) -> str:
    text = str(text or "")
    if limit <= 0 or len(text) <= limit:
        return text
    lines = text.splitlines()
    selected: Set[int] = set()
    context = max(0, _env_int("STEP2_PATCH_SECTION_SNIPPET_CONTEXT", 5))
    for idx, line in enumerate(lines):
        if _PATCH_REPAIR_LINE_RE.search(line):
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            selected.update(range(start, end))
    if not selected:
        head = text[: max(0, limit // 2)]
        tail = text[-max(0, limit // 2) :]
        return head + "\n...[truncated long section text]...\n" + tail

    chunks: List[str] = []
    previous = -2
    used = 0
    for idx in sorted(selected):
        if idx != previous + 1 and chunks:
            marker = "\n...[section text omitted]...\n"
            if used + len(marker) > limit:
                break
            chunks.append(marker)
            used += len(marker)
        line = lines[idx]
        add_len = len(line) + 1
        if used + add_len > limit:
            break
        chunks.append(line)
        used += add_len
        previous = idx
    compact = "\n".join(chunks).strip()
    if not compact:
        return text[:limit]
    return compact + "\n...[truncated to repair-relevant lines]..."


# Verbatim source sections are kept whole (see step2_format_graph_builder.LOSSLESS_SOURCE_METHODS).
LOSSLESS_SOURCE_METHODS = ('pdf_text_pages_v1', 'spec_text_sections_v1')


def _compact_section_for_patch(section: Dict[str, Any]) -> Dict[str, Any]:
    # These are already bounded, lossless chunks with page/content hashes.
    # A second keyword/head-tail compression would invalidate source coverage.
    if section.get("extraction_method") in LOSSLESS_SOURCE_METHODS:
        return copy.deepcopy(section)
    section_limit = max(512, _env_int("STEP2_PATCH_SECTION_TEXT_LIMIT", 2000))
    page_limit = max(256, _env_int("STEP2_PATCH_PAGE_TEXT_LIMIT", 1600))
    compacted = copy.deepcopy(section)
    for key in _PATCH_SECTION_TEXT_KEYS:
        value = compacted.get(key)
        if isinstance(value, str):
            compacted[key] = _compact_repair_text(value, limit=section_limit)
    pages = compacted.get("pages")
    if isinstance(pages, list):
        compacted_pages = []
        for page in pages[: max(1, _env_int("STEP2_PATCH_MAX_PAGES", 8))]:
            if isinstance(page, dict):
                page_copy = copy.deepcopy(page)
                if isinstance(page_copy.get("text"), str):
                    page_copy["text"] = _compact_repair_text(page_copy["text"], limit=page_limit)
                compacted_pages.append(page_copy)
            else:
                compacted_pages.append(page)
        if len(pages) > len(compacted_pages):
            compacted_pages.append({"omitted_pages": len(pages) - len(compacted_pages)})
        compacted["pages"] = compacted_pages
    return compacted


def summarize_sections_for_patch(
    sections,
    batch_start: int,
    batch_size: int,
    focus_only: bool = False,
):
    """Return focused sections for patch prompts.

    When focus_only is True, only the target batch range is returned with
    attention markers. Otherwise defaults to highlighting the focused range
    but keeps surrounding context.
    """
    end = batch_start + batch_size
    subset = []
    for index, section in enumerate(sections):
        # Accept both dict and plain-string sections; coerce to dict for safety.
        if not isinstance(section, dict):
            section = {"content": str(section)}
        elif "content" not in section and "summary" in section:
            # Fallback: promote summary to content if no content provided
            section = {**section, "content": section.get("summary", "")}
        if batch_start <= index < end:
            content = section.get('content', '')
            marked = _compact_section_for_patch(section)
            marked['content'] = f"<attention priority=\"high\">\n{content}\n</attention>"
            if section.get("extraction_method") not in LOSSLESS_SOURCE_METHODS:
                marked['content'] = _compact_repair_text(marked['content'], limit=max(512, _env_int("STEP2_PATCH_SECTION_TEXT_LIMIT", 2000)))
            marked['is_focused'] = True
            subset.append(marked)
    return subset


def build_initial_tree_message(sections, validation_feedback: Optional[str] = None):
  """Build the initial protocol tree message"""
  
  # Build section summaries to keep the prompt size manageable
  enhanced_sections = []
  summary_lines = []
  for section in sections:  # Use every section summary to construct the full initial tree
      if not isinstance(section, dict):
          section = {"content": str(section)}
      elif "content" not in section and "summary" in section:
          section = {**section, "content": section.get("summary", "")}
      summary = section.get('summary') or section.get('content', '')
      if isinstance(summary, str):
          summary_text = summary.strip()
      else:
          summary_text = ''

      enhanced_section = {
          'number': section.get('number', ''),
          'title': section.get('title', ''),
          'source_file': section.get('source_file', 'unknown'),
          'summary': summary_text,
          'content': section.get('content', '')[:],
          'packet_formats': section.get('packet_formats', [])[:],
      }
      enhanced_sections.append(enhanced_section)

      section_label = section.get('number') or section.get('title') or 'Section'
      if summary_text:
          summary_lines.append(f"- {section_label}: {summary_text}")
      else:
          content_preview = section.get('content', '')
          if content_preview:
              summary_lines.append(f"- {section_label}: {content_preview[:200]}")

  overall_summary = "\n".join(summary_lines) if summary_lines else "(No section summaries available.)"

  prompt = """

## CRITICAL NON-NEGOTIABLE GUIDELINES

- **Selector coverage must be exhaustive**: enumerate every documented value or range for each control field before creating variants. Keep selector constraints (enum/range) identical to the union of your `condition_on` predicates so no value is left without a variant.
- **condition_on formulas must be valid booleans**: NEVER concatenate clauses (e.g., `A (B)` or `A B`). If you need multiple cases, use explicit `or`/`and` with parentheses.
- **Represent exception patterns explicitly**: when the specification describes bit-derived or offset-based exceptions (e.g., high bit indicates error, value + constant), create separate variants with matching predicates.
- **Length bindings must respect the spec**: if a length field includes shared headers or auxiliary bytes, subtract them before binding to downstream payloads. Use `length_of` edges instead of embedding arithmetic in `size_bits`.
- **When mirroring a `length_of` formula into `size_bits`, copy the exact expression (including any offsets) and keep the `length_of` edge.** Never simplify `val(x) - 1` to `val(x)` or drop the edge; both must remain consistent with the protocol math.
- **Variants contain only selector-specific bodies**: do not copy shared headers/selectors into variants; start variant `bit_start` immediately after the shared fields to avoid overlap.
- **SMT-safe expressions are mandatory**: follow the numeric-expression and constraint rules below so every formula is Z3 compatible.
- **If the current tree already matches the specification** and no meaningful structural change remains, return an empty patch (`{}`) instead of inventing arbitrary updates.

## UNIVERSAL NODE TYPES (Protocol-Agnostic)

```yaml
protocol:     # Root container for entire protocol message
header:       # Fixed-structure container with sub-fields  
field:        # Concrete data field (leaf node)
selector:     # Control field that determines message variants
variant:      # Alternative message structure based on selector
payload:      # Variable-content data section
container:    # Generic grouping of related fields
```

## POSITIONING STRATEGY (Universal)

CRITICAL: Always Use Numeric Node IDs in Expressions

### PROHIBITED Positioning Patterns:
**Self-references**: `node.bit_start + node.size_bits` (e.g., "22.bit_start + 22.size_bits")
**Parent container references**: Child referencing its direct parent container (e.g., Node 41 with parent_id=40 using "40.bit_start + 0")
**Forward references**: Referencing nodes that don't exist or come later in sequence
**Cross-container references** without proper hierarchy
**Circular dependencies** between nodes

**Key Principle**: bit_start = parent.bit_start + offset_within_parent

## SMT-COMPATIBLE EXPRESSION RULES (STRICT)

- `bit_start` / `size_bits` expressions must remain numeric. Use only literals, additions, or `val(NODE_ID) * constant` forms.
- NEVER embed boolean expressions inside arithmetic (no `(val(X) == 2)` inside `size_bits`).
- Conditions must remain boolean expressions (comparisons + `And`/`Or`/`Not`).
- Reference other nodes via `val(NODE_ID)` or `NODE_ID.bit_start/size_bits`; do not introduce field names.
- For length prefixes, express the relationship via `length_of` edges rather than inline conditional arithmetic.

## MESSAGE TYPE CLASSIFICATION (Mandatory)

Every node MUST have exactly one message_type:

- **"bidirectional"**: Used in both directions (headers, control fields)
- **"request"**: Client-to-server/caller-to-callee structures  
- **"response"**: Server-to-client/callee-to-caller structures

## CONTROL FIELD DETECTION & VARIANT CREATION
When you find a field that determines message structure:

1. **Identify the selector field** (command, type, opcode, operation, etc.).
2. **Enumerate every documented value or range** and determine which directions (request/response/exception) exist for each.
3. **Do not duplicate selector fields** inside variants—variants only contain the selector-specific payload.
4. **Add condition_on edges** that mirror the selector constraints exactly, including any bitwise predicates required for exceptions.
5. **Verify coverage**: the union of all condition predicates must equal the selector's declared domain.
```

### ANTI-PATTERN - Avoid This:
```json
{
  "node_id": "variant1_id",
  "name": "Command_A_Variant",
  "bit_start": "same_as_selector",  // POSITION CONFLICT!
  "children_ids": ["duplicate_selector", "field1", "field2"]
},
{
  "node_id": "duplicate_selector", 
  "name": "Command_A_Control_Field",  // DUPLICATE SELECTOR!
  "bit_start": "same_as_original_selector"
}
```

**Universal Rules for Variant Creation:**
- **One selector per protocol layer** - never duplicate control fields
- **Variants contain only command-specific data** - no repeated headers/selectors
- **Sequential positioning** - variants start after shared fields
- **Conditional edges** - link selector values to appropriate data variants and cover the entire selector domain
 
**Initial tree scope:** Build a stable universal skeleton—root node, shared header nodes, base message/data containers, and essential fields only. Leave protocol-specific command or exception branches for the refinement stage.

## CONSTRAINT EXTRACTION (SMT-Compatible Only)

Extract constraints as PURE mathematical expressions:
- "must be 0" → `"enum: 0"`
- "values 1-255" → `"range: 1..255"`
- "one of A, B, C" → `"enum: A|B|C"`
- "length field maximum 64 (bytes)" → `"range: 0..64"`
- "payload maximum 64 bytes" → `"formula: size_bits >= 0 and size_bits <= 512"`  *(or omit and rely on `length_of`)*
- "aligned to 2 bytes" → `"formula: value % 2 == 0"`
- "multiple of 8" → `"formula: value % 8 == 0"`
- "depends on field X" → `"formula: value = val(X_NODE_ID)"`
- **Formula assignment**: Every `formula:` constraint must be written as `formula: value = <expression>`, for example `formula: value = min(val(37), 252 * 8)`.

**CRITICAL: SMT Solver Compatibility**
All constraints will be processed by an SMT solver for traffic generation.
FORBIDDEN constraint formats (natural language):
- "encoding: hexadecimal"
- "alignment: 2-byte boundary"
- "padding: zero bits"
- "endianness: big"

REQUIRED constraint formats (pure math only):
- Use "range: min..max" for scalar value ranges
- Use "enum: val1|val2|val3" for specific values
- Use "formula: mathematical_expression" for complex conditions
- Always use val(NODE_ID) to reference other nodes, NEVER field names

CRITICAL: NO natural language in constraints - only mathematical expressions!

## EDGE MESSAGE TYPE RULES

Every edge MUST have a message_type attribute:
- **"bidirectional"**: Edge applies to both request and response messages
- **"request"**: Edge only applies in request messages
- **"response"**: Edge only applies in response messages

For condition_on edges:
- If the condition determines a request variant → message_type = "request"
- If the condition determines a response variant → message_type = "response"
- If the condition applies to both → message_type = "bidirectional"

For length_of, offset_of, crc_of edges:
- Usually message_type = "bidirectional" unless specific to one direction
- `src` / `dst` MUST be existing numeric node IDs (write them as numbers, not quoted strings).

## OUTPUT SCHEMA (Strict JSON)

## TLV-SEQUENCE MODELLING (protocol-agnostic)

When the tree contains a repeated TLV list (Code/Tag + optional Length + Value repeated until end-of-list),
prefer modelling it as a sequence instead of many overlapping siblings:
- Create a node with `node_type: "tlv_seq"` for the repeated list.
- The `tlv_seq` node's children define the item template and will be parsed repeatedly.
- Use `stop_condition` on the tlv_seq node (e.g., `val(CODE_ID) == END_CODE`) and/or a bounded `size_bits`
  on the tlv_seq to stop at the enclosing container boundary.
- Inside each item, use `selector` + `variant` with `condition_on` edges to model PAD/END/normal items.
  IMPORTANT: `parent_id`/`children_ids` represent physical containment. Variants MUST NOT be children of the selector
  node unless they are sub-bitfields within the selector byte. Prefer an item container under `tlv_seq` and attach the
  tag selector + all variants as siblings under that item container.


{
  "protocol_tree": {
    "root_node_id": 0,
    "nodes": [
      {
        "node_id": 0,
        "name": "Protocol_Root",
        "node_type": "protocol|header|field|selector|variant|payload|container|tlv_seq",
        "message_type": "bidirectional|request|response",
        "bit_start": "positioning_expression", (parent_node_id.bit_start + offset [It can be expressed using the size of a close previous brother or accurate absolute value]) **IMPORTANT, the first item must rely on parent node bit_start**
        "size_bits": "size_expression_or_integer",
        "data_type": "binary|uint8|uint16|uint32|uint64|string|bytes",
        "byte_order": "big|little",
        "parent_id": "parent_node_id_or_null",
        "children_ids": ["array_of_child_node_ids"],
        "constraints": ["constraint_strings"],
        "source": "documentation_reference",
        "dependencies": []
      }
    ],
    "edges": [
      {
        "src": source_node_id,
        "dst": destination_node_id,
        "rel": "length_of|condition_on|offset_of|repeat_count|crc_of",
        "formula": "relationship_expression",
        "message_type": "bidirectional|request|response"
      }
    ]
  }
}


## EXAMPLE OUTPUT (Generic Protocol)


{
  "protocol_tree": {
    "root_node_id": "root",
    "nodes": [
      {
        "node_id": "root",
        "name": "Generic PDU",
        "node_type": "protocol",
        "message_type": "bidirectional",
        "bit_start": 0,
        "size_bits": "variable",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": null,
        "children_ids": ["10","20","30"],
        "constraints": [],
        "source": "Section 1.1.1",
        "dependencies": []
      },

      /* — Header — */
      {
        "node_id": "10",
        "name": "Header",
        "node_type": "header",
        "message_type": "bidirectional",
        "bit_start": root.bit_start + 0,
        "size_bits": "8 + 12.size_bits",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": "root",
        "children_ids": ["11","12"],
        "constraints": [],
        "source": "Section 2.5.8",
        "dependencies": []
      },
      {
        "node_id": "11",
        "name": "Control_Byte",
        "node_type": "field",
        "message_type": "bidirectional",
        "bit_start": "10.bit_start",
        "size_bits": 8,
        "data_type": "uint8",
        "byte_order": "big",
        "parent_id": "10",
        "children_ids": [],
        "constraints": ["enum: val1|val2|val3"],
        "source": "Section 1.1.2",
        "dependencies": []
      },
      {
        "node_id": "12",
        "name": "Length_Field",
        "node_type": "field",
        "message_type": "bidirectional",
        "bit_start": "10.bit_start + 11.size_bits",
        "size_bits": "varint",
        "data_type": "uint",
        "byte_order": "big",
        "parent_id": "10",
        "children_ids": [],
        "constraints": [],
        "source": "Section 2.1.3",
        "dependencies": []
      },

      /* — Body Section A — */
      {
        "node_id": "20",
        "name": "Body_SectionA",
        "node_type": "header",
        "message_type": "bidirectional",
        "bit_start": "root.bit_start + 10.size_bits",
        "size_bits": "val(12)",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": "root",
        "children_ids": [],
        "constraints": [],
        "source": "Section 3.1.6",
        "dependencies": []
      },

      /* — Body Section B — */
      {
        "node_id": "30",
        "name": "Body_SectionB",
        "node_type": "payload",
        "message_type": "bidirectional",
        "bit_start": "root.bit_start + 10.size_bits",
        "size_bits": "val(12)",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": "root",
        "children_ids": [],
        "constraints": [],
        "source": "Section 1.1.5",
        "dependencies": []
      }
    ],
    "edges": [
      {
        "src": 12,
        "dst": 20,
        "rel": "length_of",
        "formula": "val(12)",
        "message_type": "bidirectional"
      },
      {
        "src": 12,
        "dst": 30,
        "rel": "length_of",
        "formula": "val(12)",
        "message_type": "bidirectional"
      },
      {
        "src": 11,
        "dst": 20,
        "rel": "condition_on",
        "formula": "val(11) == value1",
        "message_type": "request"
      },
      {
        "src": 11,
        "dst": 30,
        "rel": "condition_on",
        "formula": "val(11) == value2",
        "message_type": "response"
      }
    ]
  }
}

## CRITICAL SUCCESS FACTORS

1. **Protocol Agnostic**: No hardcoded assumptions about specific protocols
2. **Structure First**: Focus on binary layout over semantic meaning
3. **Concrete Only**: Extract only fields with clear bit positions/sizes
4. **Relationship Driven**: Use edges to express field interdependencies
5. **Variant Aware**: Detect and model conditional message structures
6. **Position Safe**: Avoid circular dependencies and cross-container references
7. **Hierarchy Complete**: Whenever you add a node, append its ID to the parent `children_ids` so the tree stays connected.

## QUALITY CHECKLIST

Before returning your result, verify:
- [ ] Every node has unique numeric node_id
- [ ] Every node has required message_type field
- [ ] No self-referential positioning expressions
- [ ] All selector fields have corresponding variant nodes
- [ ] Container hierarchy makes logical sense
- [ ] Size and position expressions are well-formed

Build a **minimal but correct** structure. Better to have a simple, accurate skeleton than a complex, incorrect tree.

## DOCUMENT SUMMARY (GLOBAL CONTEXT):
""" + overall_summary + """

## DOCUMENTATION TO ANALYZE:
""" + json.dumps(enhanced_sections, indent=2) + """

Please analyze the above documentation and extract a protocol tree structure following the guidelines above.
Output ONLY the JSON object, no other text."""

  if validation_feedback:
      prompt += "\n\n## VALIDATION FEEDBACK FROM PREVIOUS ATTEMPT\n" + validation_feedback.strip() + "\n\nPlease address the validation issues above when generating the protocol tree."

  message = [
      {"role": "system", "content": "You are building a protocol format tree. Be specific and comprehensive. CRITICAL: You must respond with valid JSON only. No markdown, no comments, no extra text."},
      {"role": "user", "content": prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no markdown formatting, no comments."}
  ]
  return message

def build_patch_refinement_message(
    tree,
    marked_sections,
    batch_start,
    batch_size,
    feedback: str | None = None,
    previous_patch: str | None = None,
    previous_attempt: dict | None = None,
    *,
    mode: str = "fix",
    avoid_summaries: List[str] | None = None,
    experience: Optional[List[Dict[str, Any]]] = None,
):
    """Compose the refinement prompt with structured context and instructions."""
    import json

    def _strict_validator_loop_enabled() -> bool:
        raw = os.getenv("STEP2_STRICT_VALIDATOR_LOOP", "")
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _strip_hex_preview_from_issue_description(desc: str) -> str:
        """
        Remove hex previews injected into Issue.description examples (pkt#.. hex=...).

        In strict traffic_fix mode we only want packet_hex samples from TrafficParseFailure groups
        to appear in the LLM prompt, to avoid duplicated traffic bytes and reduce prompt noise.
        """
        if not desc:
            return desc
        # Replace "hex=...." (hex string up to next delimiter) with nothing.
        # Keep pkt indices so examples remain identifiable.
        desc = re.sub(r"\s+hex=[0-9a-fA-F]+", "", desc)
        # Clean up doubled spaces that can result from removal.
        desc = re.sub(r"\s{2,}", " ", desc).strip()
        return desc

    tree_json = json.dumps(tree, indent=2)
    sections_json = json.dumps(marked_sections, indent=2)
    pending_lines: List[str] = []
    recent_lines: List[str] = []
    validator_lines: List[str] = []
    followup_lines: List[str] = []
    issues_payload = []
    records_payload = []
    validator_errors = []
    validator_extras = []
    avoid_block = ""
    if avoid_summaries:
        unique_avoids = list(dict.fromkeys(avoid_summaries))[-8:]
        lines = [
            "### STRATEGY CONSTRAINTS (AVOID REPETITION)",
            "The following patch strategies have ALREADY been tried at this state (as siblings) and were rejected.",
            "You MUST generate a DIFFERENT approach (e.g., if reparenting failed, try fixing formulas; if formulas failed, try restructuring).",
            "DO NOT output the same patch intent again:",
        ]
        for s in unique_avoids:
            lines.append(f"- [AVOID] {s}")
        avoid_block = "\n".join(lines) + "\n"

    if isinstance(previous_attempt, dict):
        issues_payload = previous_attempt.get("pending_issues") or []
        records_payload = previous_attempt.get("recent_records") or []
        validator_errors = previous_attempt.get("validator_errors") or []
        validator_extras = previous_attempt.get("validator_extras") or []

        variant_warnings = [
            str(w) for w in validator_extras 
            if "variant condition" in str(w).lower() and "cannot be satisfied" in str(w).lower()
        ]

        if variant_warnings:
            guidance_msg = (
                "CRITICAL PRIORITY: The validator found unreachable variants (condition cannot be satisfied). "
                "You MUST fix the 'condition_on' edges or the selector's constraints so these variants are reachable. "
                "Do NOT attempt to fix missing children or length errors inside a variant until its condition is valid."
            )
            followup_lines.append(f"- {guidance_msg}")
            
            for w in variant_warnings[:3]:
                followup_lines.append(f"  * Focus on fixing this warning: {w}")
            
            action_msg = (
                "REQUIRED ACTION: Expand the parent selector's 'enum'/'range' constraints to include the value required "
                "by the variant, OR correct the 'condition_on' formula if it mismatches the selector."
            )
            followup_lines.append(f"- {action_msg}")

        followup_tasks = previous_attempt.get("followup_tasks") or []
        for task in followup_tasks[:10]:
            followup_lines.append(f"- {task}")
    node_map, depth_map = _build_node_maps(tree)
    # Sort issues by priority: parent/earlier nodes first, then higher hits/severity
    sorted_issues = sorted(issues_payload[:12], key=lambda x: _issue_priority(x, node_map, depth_map))
    for issue in sorted_issues:
        if not isinstance(issue, dict):
            continue
        identifier = issue.get("id")
        issue_type = issue.get("type")
        severity = issue.get("severity")
        description = issue.get("description") or "Unspecified validator issue"
        if _strict_validator_loop_enabled() and (mode or "").strip().lower() == "traffic_fix":
            description = _strip_hex_preview_from_issue_description(str(description))
        target = issue.get("target")
        detail = description
        if target:
            detail += f" (target: {target})"
        label = f"[{severity}] {issue_type}" if severity or issue_type else "ISSUE"
        if identifier:
            pending_lines.append(f"- {identifier} {label}: {detail}")
        else:
            pending_lines.append(f"- {label}: {detail}")
    for record in records_payload[:5]:
        if not isinstance(record, dict):
            continue
        summary = record.get("summary") or "(no summary)"
        reward = record.get("reward")
        introduced = record.get("introduced") or []
        resolved = record.get("resolved") or []
        line = f"- {record.get('hash', '')[:8]} {summary}"
        if reward is not None:
            line += f" | reward={reward:.2f}"
        if introduced or resolved:
            line += f" | introduced={len(introduced)} resolved={len(resolved)}"
        recent_lines.append(line)
    for err in validator_errors[:5]:
        validator_lines.append(f"- ERROR: {err}")
    for extra in validator_extras[:5]:
        validator_lines.append(f"- NOTE: {extra}")

    experience_lines: List[str] = []
    if experience:
        for entry in experience:
            packet_idx = entry.get("packet_idx", "?")
            error = entry.get("error") or entry.get("issue") or "unknown issue"
            action = entry.get("action") or entry.get("strategy") or entry.get("summary") or ""
            experience_lines.append(f"- Packet #{packet_idx}: {error} -> {action}".strip())
    experience_block = "### EXPERIENCE FROM PREVIOUS PACKETS\n" + ("\n".join(experience_lines) if experience_lines else "None") + "\n"

    pending_block = "### PENDING ISSUES\n" + ("\n".join(pending_lines) if pending_lines else "None") + "\n"
    recent_block = "### RECENT PATCHES\n" + ("\n".join(recent_lines) if recent_lines else "None") + "\n"
    validator_block = "### VALIDATOR NOTES\n" + ("\n".join(validator_lines) if validator_lines else "None") + "\n"
    followup_block = "### TARGETED FIX GUIDANCE\n" + ("\n".join(followup_lines) if followup_lines else "None") + "\n"
    feedback_block = ""
    if feedback:
        feedback_block = "### FEEDBACK FROM PREVIOUS ITERATION\n" + feedback.strip() + "\n"
    format_warning_block = ""

    normalized_mode = (mode or "fix").strip().lower()
    prompt_context = tree.get("_prompt_context") if isinstance(tree, dict) else None
    if isinstance(prompt_context, dict) and prompt_context.get("type") == "doc_structure_compact":
        issue_examples_block = build_issue_example_block(issues_payload, validator_errors, validator_extras)
        focus_issue_ids = prompt_context.get("focused_issue_ids") or []
        if isinstance(focus_issue_ids, list) and focus_issue_ids:
            focus_line = "- Resolve exactly these focused structure issues: " + ", ".join(str(item) for item in focus_issue_ids) + "\n"
        else:
            focus_line = "- Resolve the listed documentation structure issue with one small local patch.\n"
        patch_skeleton = prompt_context.get("patch_skeleton")
        patch_skeleton_json = json.dumps(patch_skeleton, indent=2) if isinstance(patch_skeleton, dict) else "{}"
        compact_objective_block = (
            "### CURRENT OBJECTIVE\n"
            f"{focus_line}"
            "- Use the safe structural repair skeleton when it targets the shown variant or existing node.\n"
            "- For missing scoped fields, add child fields under the listed variant and update that variant's children_ids.\n"
            "- For weak name alignment, rename/update the existing node; do not create a duplicate wire field.\n"
            "- If the safe repair is ambiguous or would attach fields to the wrong selector scope, return {}.\n"
        )
        compact_patch_format = (
            "{\n"
            "  \"patch_metadata\": {\"description\": \"...\", \"intent\": \"Doc structure alignment\"},\n"
            "  \"new_nodes\": [\n"
            "    {\"node_id\": 153, \"name\": \"Documented_Field\", \"node_type\": \"field\", \"parent_id\": 38, \"children_ids\": [], \"size_bits\": 8, \"constraints\": [\"enum: 0\"]}\n"
            "  ],\n"
            "  \"nodes_to_remove\": [],\n"
            "  \"node_updates\": [\n"
            "    {\"node_id\": 38, \"property\": \"children_ids\", \"value\": [1, 2, 153], \"reason\": \"Attach documented child field\"}\n"
            "  ],\n"
            "  \"new_edges\": [],\n"
            "  \"edge_updates\": [],\n"
            "  \"edge_removes\": [],\n"
            "  \"validation_notes\": []\n"
            "}\n"
        )
        compact_prompt = """## DOC-STRUCTURE COMPACT PATCH

### OUTPUT FORMAT
- Return exactly one ASCII JSON object. If no safe local patch exists, return {{}}.
- The top-level JSON must be an object. Do not return prose, markdown, a bare array, or comments.
- This prompt shows a partial tree neighborhood only. Node IDs are real and refer to the full tree.
- Patch only shown real node IDs, except for new node IDs already proposed in the skeleton.
- Preserve existing children, constraints, node IDs, and edges unless the safe repair explicitly updates them.
- Do not move fields across selector scopes. Do not add fields from one selector value's row to a sibling variant.

### STRUCTURE RULES
- A missing on-wire field must be represented as a child node; a variant-level formula alone is not enough.
- When adding child fields, update the parent variant/container `children_ids` and keep existing child IDs.
- New child fields must be layout-valid: the first child starts at `<parent_id>.bit_start + 0`; each later child starts at `<previous_child_id>.bit_start + <previous_child_id>.size_bits`.
- If adding a child to a fixed-size parent, also update that parent `size_bits` so all listed children fit exactly, or return {{}}.
- Never place a child before its parent start; never add a child whose end exceeds the parent size.
- Use documented `size_bits` and `direct_constraint` from the safe repair when present.
- If a variant already has a generic variable payload for the same bytes, prefer adding a local structured child only when the validator safe repair explicitly points to that variant.
- If the missing field has `size_bits: "variable"` and the parent already has children, return {{}} unless the skeleton includes a complete safe rewrite for all affected siblings.
- If the evidence or shown tree suggests the safe repair points to the wrong selector scope, return {{}} rather than damaging another variant.

### COMPACT TREE CONTEXT
{tree_json}

{objective_block}{pending_block}
{recent_block}
{validator_block}
{followup_block}{feedback_block}
{avoid_block}
### DOCUMENTATION / STRUCTURE EVIDENCE
{sections_json}

{issue_examples_block}
### FILL THIS PATCH SKELETON
Use this object shape. Keep the top-level object keys.
{patch_skeleton_json}

### PATCH SCHEMA
{compact_patch_format}
"""
        prompt = compact_prompt.format(
            tree_json=tree_json,
            objective_block=compact_objective_block,
            pending_block=pending_block,
            recent_block=recent_block,
            validator_block=validator_block,
            followup_block=followup_block,
            feedback_block=feedback_block + format_warning_block,
            avoid_block=avoid_block,
            sections_json=sections_json,
            issue_examples_block=("\n" + issue_examples_block if issue_examples_block else ""),
            patch_skeleton_json=patch_skeleton_json,
            compact_patch_format=compact_patch_format,
        )
        return [
            {
                "role": "system",
                "content": "You are a protocol tree structure patch generator. Generate valid ASCII JSON patches only. If no safe local patch exists, return {}.",
            },
            {"role": "user", "content": prompt},
        ]

    if isinstance(prompt_context, dict) and prompt_context.get("type") == "doc_constraint_compact":
        issue_examples_block = build_issue_example_block(issues_payload, validator_errors, validator_extras)
        focus_issue_ids = prompt_context.get("focused_issue_ids") or []
        if isinstance(focus_issue_ids, list) and focus_issue_ids:
            focus_line = "- Resolve exactly these focused issues: " + ", ".join(str(item) for item in focus_issue_ids) + "\n"
        else:
            focus_line = "- Resolve the listed documentation constraint coverage issue with one small local patch.\n"
        patch_skeleton = prompt_context.get("patch_skeleton")
        patch_skeleton_json = json.dumps(patch_skeleton, indent=2) if isinstance(patch_skeleton, dict) else "{}"
        compact_objective_block = (
            "### CURRENT OBJECTIVE\n"
            f"{focus_line}"
            "- Use the safe repair skeleton unless it would constrain the wrong selector scope.\n"
            "- For exact selector variants, append enum/range to the listed target field.\n"
            "- For catch-all/generic variants, use a guarded formula on the variant, e.g. `formula: val(SEL) != K or (val(FIELD) == V)`.\n"
            "- Do not add condition_on edges to leaf fields and do not widen or delete shared header constraints.\n"
        )
        compact_patch_format = (
            "{\n"
            "  \"patch_metadata\": {\"description\": \"...\", \"intent\": \"Doc constraint coverage\"},\n"
            "  \"new_nodes\": [],\n"
            "  \"nodes_to_remove\": [],\n"
            "  \"node_updates\": [\n"
            "    {\"node_id\": 17, \"property\": \"constraints\", \"value\": [\"existing\", \"formula: val(5) != 129 or (val(17) == 1 or val(17) == 2)\"], \"reason\": \"...\"}\n"
            "  ],\n"
            "  \"new_edges\": [],\n"
            "  \"edge_updates\": [],\n"
            "  \"edge_removes\": [],\n"
            "  \"validation_notes\": []\n"
            "}\n"
        )
        compact_prompt = """## DOC-CONSTRAINT COMPACT PATCH

### OUTPUT FORMAT
- Return exactly one ASCII JSON object. If no safe local patch exists, return {{}}.
- The top-level JSON must be an object. Do not return a bare array, string, or constraints list.
- This prompt shows a partial tree neighborhood only. Node IDs are real and refer to the full tree.
- Patch only real node IDs shown in this compact context. Do not delete or rewrite unseen nodes.
- Preserve existing constraints: when updating a `constraints` list, include all existing entries plus the new enum/range/formula.
- If the skeleton uses a guarded formula, keep the implication form; do not replace it with an unconditional leaf enum/range.

### FORMULA RULES
- Use only `val(<ID>)`, integers, and `+ - * / % ( ) and or not == != >= > <= <`.
- No Unicode symbols, no chained comparisons, no `&&` or `||`.
- For scoped implications use `formula: val(SEL) != K or (val(FIELD) == V1 or val(FIELD) == V2)`.
- For scoped ranges use `formula: val(SEL) != K or (val(FIELD) >= LO and val(FIELD) <= HI)`.
- Never attach a constraint documented for selector value K to a shared/global field unless that field is only present in the exact K variant.

### COMPACT TREE CONTEXT
{tree_json}

{objective_block}{pending_block}
{recent_block}
{validator_block}
{followup_block}{feedback_block}
{avoid_block}
### DOCUMENTATION / CONSTRAINT EVIDENCE
{sections_json}

{issue_examples_block}
### FILL THIS PATCH SKELETON
Use this object shape. Replace only the placeholder reason text if needed; keep the top-level object keys.
{patch_skeleton_json}

### PATCH SCHEMA
{compact_patch_format}
"""
        prompt = compact_prompt.format(
            tree_json=tree_json,
            objective_block=compact_objective_block,
            pending_block=pending_block,
            recent_block=recent_block,
            validator_block=validator_block,
            followup_block=followup_block,
            feedback_block=feedback_block + format_warning_block,
            avoid_block=avoid_block,
            sections_json=sections_json,
            issue_examples_block=("\n" + issue_examples_block if issue_examples_block else ""),
            patch_skeleton_json=patch_skeleton_json,
            compact_patch_format=compact_patch_format,
        )
        return [
            {
                "role": "system",
                "content": "You are a protocol tree doc-constraint patch generator. Generate valid ASCII JSON patches only. If no safe local patch exists, return {}.",
            },
            {"role": "user", "content": prompt},
        ]

    if isinstance(prompt_context, dict) and prompt_context.get("type") == "doc_relational_compact":
        issue_examples_block = build_issue_example_block(issues_payload, validator_errors, validator_extras)
        focus_issue_ids = prompt_context.get("focused_issue_ids") or []
        if isinstance(focus_issue_ids, list) and focus_issue_ids:
            focus_line = "- Resolve exactly this focused issue: " + ", ".join(str(item) for item in focus_issue_ids) + "\n"
        else:
            focus_line = "- Resolve the listed documentation relational obligations with one small local patch.\n"
        patch_skeleton = prompt_context.get("patch_skeleton")
        patch_skeleton_json = json.dumps(patch_skeleton, indent=2) if isinstance(patch_skeleton, dict) else "{}"
        compact_objective_block = (
            "### CURRENT OBJECTIVE\n"
            f"{focus_line}"
            "- Prefer `constraint_ir:` plus a safe formula/length/alignment binding over structure rewrites.\n"
            "- Do not try to solve other relation issues in the same response.\n"
        )
        compact_patch_format = (
            "{\n"
            "  \"patch_metadata\": {\"description\": \"...\", \"intent\": \"Doc relational obligation\"},\n"
            "  \"new_nodes\": [],\n"
            "  \"nodes_to_remove\": [],\n"
            "  \"node_updates\": [\n"
            "    {\"node_id\": 17, \"property\": \"constraints\", \"value\": [\"existing\", \"constraint_ir: {\\\"kind\\\":\\\"mod_alignment\\\",\\\"expr\\\":\\\"val(17) % 8 == 0\\\",\\\"field_ids\\\":[17]}\", \"formula: val(17) % 8 == 0\"], \"reason\": \"...\"}\n"
            "  ],\n"
            "  \"new_edges\": [\n"
            "    {\"src\": 5, \"dst\": 17, \"rel\": \"length_of\", \"formula\": \"val(5)*8\", \"message_type\": \"bidirectional\"}\n"
            "  ],\n"
            "  \"edge_updates\": [],\n"
            "  \"edge_removes\": [],\n"
            "  \"validation_notes\": []\n"
            "}\n"
        )
        compact_prompt = """## RELATION-ONLY COMPACT PATCH

### OUTPUT FORMAT
- Return exactly one ASCII JSON object. If no safe local patch exists, return {{}}.
- The top-level JSON must be an object. Do not return a bare array, string, or constraints list.
- This prompt shows a partial tree neighborhood only. Node IDs are real and refer to the full tree.
- Do not delete or rewrite unseen nodes. Prefer a local node constraint, size_bits update, length_of edge, or condition_on formula using shown IDs.
- Preserve existing constraints: when updating a `constraints` list, include all existing entries plus the new `constraint_ir: ...`; include `formula: ...` only when the formula is executable and locally safe.
- If you add only one constraint, still wrap it as `node_updates`: `[{{"node_id": <shown_id>, "property": "constraints", "value": [<all existing constraints>, "constraint_ir: {{...}}"]}}]`.
- `constraint_ir:` is a JSON object string. Use keys like `kind`, `expr`, and `field_ids`, for example `constraint_ir: {{"kind":"mutual_exclusion","expr":"not (val(3) == 1 and val(4) == 1)","field_ids":[3,4]}}`.
- `val(ID)` may reference only scalar parsed-value nodes: selector/code/type/length/checksum/fixed scalar field. Never use `val(ID)` for payload/body/container/data/variant nodes; bind formulas to selector/length/scalar children, or keep the knowledge as `constraint_ir:` without an executable formula until scalar children exist.
- For packed bit arrays with final-byte padding, use this schema constraint string on the payload/bit-array node:
  `packed_bits: unit_bits=1; order=lsb_first|msb_first|unknown; pad=zero; count=val(ID)|external`.
- Use `count=val(ID)` only when the bit-count/quantity field is shown in the same message. If the count comes from request context or another packet, use `count=external` instead of inventing a formula.

### FORMULA RULES
- Use only `val(<ID>)`, `<ID>.size_bits`, `<ID>.bit_start`, integers, and `+ - * / % ( ) and or not == != >= > <= <`.
- `val(<ID>)` is legal only for scalar parsed-value nodes. If the shown ID is a payload/body/container/data/variant node, do not reference it with `val()`; use its scalar selector/length/value child, a length_of edge, or `constraint_ir:` only.
- No Unicode symbols, no chained comparisons, no `&&` or `||`.
- For implications use `formula: val(A) != k or val(B) == m`.
- For mutual exclusion use `formula: not (val(A) == 1 and val(B) == 1)` or `formula: val(A) + val(B) <= 1`.
- For alignment use modulo, e.g. `formula: val(OFFSET_ID) % 8 == 0`.
- For length relations use a direct forward binding, e.g. `formula: val(LEN_ID) == 24 + val(AUTH_LEN_ID)` or a `length_of` edge in bits.
- For packed-bit padding rules, prefer the `packed_bits:` schema constraint above over a vague `formula: padding == 0`.
- For relation knowledge that cannot be safely executed, use only `constraint_ir:` instead of inventing a parser-side formula.

### COMPACT TREE CONTEXT
{tree_json}

{objective_block}{pending_block}
{recent_block}
{validator_block}
{followup_block}{feedback_block}
{avoid_block}
### DOCUMENTATION / RELATION EVIDENCE
{sections_json}

{issue_examples_block}
### FILL THIS PATCH SKELETON
Use this object shape. Replace only the placeholder formula/reason text as needed; keep the top-level object keys.
{patch_skeleton_json}

### PATCH SCHEMA
{compact_patch_format}
"""
        prompt = compact_prompt.format(
            tree_json=tree_json,
            objective_block=compact_objective_block,
            pending_block=pending_block,
            recent_block=recent_block,
            validator_block=validator_block,
            followup_block=followup_block,
            feedback_block=feedback_block + format_warning_block,
            avoid_block=avoid_block,
            sections_json=sections_json,
            issue_examples_block=("\n" + issue_examples_block if issue_examples_block else ""),
            patch_skeleton_json=patch_skeleton_json,
            compact_patch_format=compact_patch_format,
        )
        return [
            {
                "role": "system",
                "content": "You are a protocol tree relation patch generator. Generate valid ASCII JSON patches only. If no safe local patch exists, return {}.",
            },
            {"role": "user", "content": prompt},
        ]

    # Optional atomic mode: encourage the model to return a very small patch
    # (one focused change) so that MCTS explores multiple steps.
    # - Default: enabled for traffic_fix (safer, avoids giant structural edits).
    # - Override: STEP2_PATCH_ATOMIC_MODE=1|true to force on; =0|false to force off.
    import os as _os
    atomic_env = _os.getenv("STEP2_PATCH_ATOMIC_MODE")
    if atomic_env is None:
        atomic_mode = normalized_mode == "traffic_fix"
    else:
        atomic_mode = atomic_env.lower() in {"1", "true", "yes", "on"}

    if normalized_mode == "expand":
        objective_lines = [
            "Validator issues are currently resolved.",
            "Expand the protocol structure to cover documentation gaps, adding nodes or edges even if temporary warnings appear.",
            "Prioritise structural completeness and coverage of documented flows before returning to issue cleanup.",
        ]
        instructions = [
            'Operate on the existing tree; do not regenerate it from scratch.',
            'Focus on adding or restructuring nodes and edges to cover the documentation, even if this introduces minor validator warnings.',
            'Document every structural change in patch_metadata.description so follow-up cleanup can reference it.',
            'Avoid deleting existing coverage unless it contradicts the documentation.',
            'When deleting edges, use either edge_removes entries (e.g., {"src": "<SRC_NODE_ID>", "dst": "<DST_NODE_ID>", "rel": "length_of"}) or edge_updates entries with update_type="remove" plus edge_identifier; JSON Patch style {"op", "path"} entries are rejected.',
            'If the tree already captures the documentation and no structural expansion remains, respond with {}.',
            'Return ONLY a JSON object that matches the schema below -- no commentary.',
        ]
    else:
        objective_lines = [
            "Resolve the outstanding validator issues and follow-up tasks for this batch.",
        ]
        instructions = [
            # === NEW: Ground Truth Hierarchy Strategy ===
            '### CRITICAL: THE HIERARCHY OF TRUTH',
            '1. **Supreme Authority (Ground Truth)**: The **Documentation** text and the **Traffic Hex Samples** (in issue descriptions) are the absolute truth. Never contradict them.',
            '2. **The Validator is a tool, not a god**: The validator reports inconsistent states (Symptoms). Its suggestions (e.g., "add edge") are heuristics and can be WRONG if they contradict the Ground Truth.',
            '3. **Conflict Resolution**: ',
            '   - **IF** Validator says "Add length_of edge" **BUT** Documentation says "Fixed 2-byte field",',
            '   - **THEN** the Validator is confused because you defined `size_bits="variable"`.',
            '   - **ACTION**: Do NOT add the edge. Instead, **FIX the attribute** (set `size_bits=16`). This aligns with Ground Truth AND silences the Validator.',
            '4. **Evidence-Based Reasoning**: Before every fix, look at the `hex=...` sample. Count the bytes. If the tree structure does not match the bytes, the tree is wrong, regardless of what the validator says.',
            # ============================================
            # === NEW: Causal Dependency Rule ===
            '### CRITICAL: NO FORWARD REFERENCES',
            '1. **Time Flow**: A node can ONLY reference nodes that appear **before** it in the stream.',
            '2. **Prohibited**: Do NOT define a field\'s size based on a node that comes AFTER it (e.g., do not size a Length field based on the Payload).',
            '3. **Fix**: If you need to relate Length and Payload, the dependency MUST go from Predecessor (Length) -> Successor (Payload).',
            # ===================================
            'Formula syntax (validator/interpreter/Z3 share the SAME rules):',
            '  - Allowed identifiers: val(<ID>), <ID>.size_bits, <ID>.bit_start. Do NOT invent new symbols (e.g., size_bits_1).',
            '  - Operators: + - * / ( ) and/or/not with == != >= > <= <. NO &&, ||, &, |.',
            '  - No chained comparisons: write `val(x) >= 1 and val(x) <= 4` (NOT `1 <= val(x) <= 4`).',
            '  - OR must be fully expanded: GOOD `val(7)==1 or val(7)==2 or val(7)==4 or val(7)==15`; BAD `val(7)==1 || 2 || 4 || 15` (forbidden).',
            '  - size_bits must be pure arithmetic (no booleans); condition_on must be pure selector predicates (no message_type checks or other fields).',
            '  - condition_on scope rule: for an edge src->dst with rel="condition_on", the formula may reference ONLY val(src) (plus constants/boolean ops). Referencing val(other_id) is INVALID; if multiple control fields are involved, introduce nested selector/variant routing instead of mixing them in one condition.',
            '  - Avoid shadow selectors: do NOT create per-variant selector copies like Type_A/Type_B and then gate variants on those. Create ONE shared selector field (Type/Opcode) at the correct wire offset, and route mutually-exclusive variants from it.',
            '  - Lengths, size_bits and length_of are all in bits: if a count field counts bytes use val(<field>)*8, and if it already counts bits use val(<field>) directly; do not write inverse or rounding formulas.',
            '  - For nodes with `data_type=\"bytes\"` or `node_type=\"payload\"`, never express a length with `range: ... <= value <= ...` (bytes are compared as one large integer). Bound the length with `size_bits` (e.g. `formula: size_bits >= 8 and size_bits <= 128`, in bits) or rely on length_of/size_bits expressions only.',
            '  - A length_of edge goes from the length/count field to the payload, and src/dst message_type must agree; put range checks in the constraints of the length field, not in length_of.',
            '  - length_of and constraints must not use division, rounding, or inverse forms such as (val(x)+7)/8; use forward numeric formulas and, where needed, add a simple range/enum constraint to the count field.',
            '  - A parent container\'s size_bits should equal the sum of its children or be bound by length_of; do not use max(variant.size_bits) or a hard-coded constant to cover a variable child; every variable child must have a length_of binding.',
            '  - Avoid two equivalent payloads at the same level (such as *_Bytes next to *_Response); keep the one whose length chain is clear.',
            '  - Reserved/Padding/Spare/Unused fields (from tables or wire-format descriptions) occupy space on the wire: do not omit them because they seem to carry no meaning; a missing one shifts every later field (gap_bits / incomplete parse / read past end). Add the missing placeholder field instead of moving later fields to make the alignment fit.',
            '  - `reserved_values` describes reserved or unassigned values; it is not a hard constraint that the value must fall in that range: do not turn it into a `range:` constraint, and do not combine it with `enum:` into mutually exclusive constraints (e.g. enum=0..8 + range=9..31). Add a hard constraint such as `==0` / `range 0..0` only when the specification says MUST/SHALL (for example MUST be zero).',
            '  - Forward/self references are prohibited; only reference already-parsed predecessors.',
            '  - Edges must not be self-referential (src != dst), especially for length_of/condition_on.',
            '  - Selector routing must target variants (not fields): if a node is node_type="selector", its outgoing condition_on edges MUST point to node_type="variant" containers (never directly to leaf field nodes). Group all fields that share the same selector predicate under ONE variant container and attach a single condition_on edge to that variant.',
            '  - Selector variants are siblings of the selector leaf under its enclosing container. Do not parent a variant under the selector leaf itself; the selector leaf keeps only its own wire size.',
            '  - Each variant must have exactly ONE condition_on from a single selector; do not attach multiple selectors or duplicate formulas to the same variant. Formulas across variants must be mutually exclusive. If you cannot express a clean single condition, return {} instead of emitting multiple edges.',
            '  - Numeric selector labels: if documentation writes `22 (0x16)` or `0x16 (22)`, the selector value is one integer with decimal value 22. Use `val(selector)==22`; do not create a misleading `OP16` name unless the documented decimal value is actually 16. Prefer names like `OP22_0x16` when both forms are shown.',
            'Constraint string format: use only ASCII. Allowed executable forms are `enum: 1|2|3`, `range: 1..4`, and `formula: val(7) >= 1 and val(7) <= 4`; relation-only protocol knowledge may use `constraint_ir: {"kind":"...","expr":"...","field_ids":[...]}`. Do NOT use Unicode math/arrow symbols or chained comparisons.',
            'If you violate these syntax rules, your patch is invalid—rewrite the formulas into the allowed form.',
            '### FIX PRIORITY',
            'When applying patches, always follow this priority order:',
            '1. MUST NOT introduce any new issues with severity ERROR.',
            '2. MUST prioritize fixing existing issues with severity ERROR, especially STRUCTURE and SEMANTICS issues.',
            '3. Only attempt to fix WARN / HINT issues if this does not introduce any new ERROR and the change is local/simple.',
            '4. If you cannot fix all issues in one patch, prefer fixing some ERRORs and leave WARNs unresolved rather than risking new ERRORs.',
            'Treat the current tree as a draft that may contain structural mistakes. Documentation and validator feedback are the source of truth.',
            'It is acceptable to move, merge, or delete nodes when the documentation/validator indicates duplicate headers, mis-parented children, or missing containers.',
            'Maintain a single logical root (typically node 0). If validator reports "Multiple root nodes", FIRST reattach the stray node by setting its parent_id and updating the correct parent.children_ids; only delete a root candidate if the documentation clearly says it should not exist.',
            'For selector/variant conflicts (Invalid overlap / Non-exclusive conditions):',
            '  - Preferred: delete generic catch-all variants that lack conditions (e.g., Request_Payload/Response_Payload) and remove all references (parent.children_ids, edges).',
            '  - If keeping them: add exactly one condition_on per variant with a mutually exclusive formula covering only uncovered cases; do NOT leave unconditional branches alongside conditioned variants.',
            '  - When feedback mentions Non-exclusive conditions with <SpecificVariant>, first clean up or add conditions to the generic variants involved (such as Request/Response_Payload) instead of deleting only one or two condition_on edges.',
            '  - If overlapping siblings are alternatives selected by the same opcode/function/type field, they MUST share ONE selector at their common parent layer. Reuse that shared selector and route all variants from it; do NOT create per-variant shadow selectors.',
            '  - Forbidden overlap fix: creating Opcode_A inside Variant_A and Opcode_B inside Variant_B, then attaching separate condition_on edges from each local selector. This does NOT establish mutual exclusivity.',
            '  - Do NOT serialize true alternatives by shifting one variant after another just to remove overlap. Overlap is expected for mutually-exclusive variants controlled by the same selector.',
            'When adding/removing condition_on edges, always include src/dst/rel + formula + message_type to ensure precise matching; avoid fuzzy deletes that skip the intended edge.',
            'Any time you delete or move a node, synchronize both sides: update parent.children_ids, clear child.parent_id if detached, and remove edges pointing to removed nodes.',
            'For selector nodes, add an explicit enum/domain constraint when the documentation provides a bounded set of values. Ensure the enum is consistent with all outgoing condition_on edges (do not exclude values referenced by any branch).',
            'For documentation-structure gaps from a repeated selector/table: treat all listed gaps that share the same guard field as one coherent selector-table repair. Reuse the existing selector, add the visible missing guarded variants/fields from PENDING ISSUES, and add mutually exclusive condition_on edges for those values. Do not patch only the first row when multiple sibling rows are explicitly listed.',
            'For TLV-style tables: keep a generic TLV/Other branch for unconstrained rows, but add exact guarded variants when the documentation lists row-specific on-wire fields, fixed lengths, or fixed value constraints. The generic catch-all must exclude exact guarded variants and no-body values.',
            'For context-scoped documentation constraints (request/response/client/server/when/if): do NOT append an unconditional enum/range to a field shared by all messages. Add the constraint only inside a matching variant/message scope. If no such scope exists, leave that doc claim unresolved rather than causing traffic samples from other modes to fail.',
            'Keep the patch focused on resolving the listed pending issues and follow-up tasks; structural repairs (e.g., re-homing a selector, deduplicating fields, restoring missing length bindings) are encouraged whenever they directly address the validator feedback.',
            'Apply complete updates directly (no subaction splitting).',
            'When deleting edges, use either edge_removes entries (e.g., {"src": 18, "dst": 23, "rel": "length_of"}) or edge_updates entries with update_type="remove" plus edge_identifier; JSON Patch style {"op", "path"} entries are rejected.',
            'Example edge identifiers shown below are placeholders. Always confirm the correct source/destination IDs from the current tree and validator feedback before deleting a binding, especially for Byte Count to payload edges.',
            'Preserve numeric expressions and message typing rules when updating nodes or edges.',
            'If the tree already reflects the documentation and no further fixes are needed, respond with {}.',
            'Return ONLY a JSON object that matches the schema below -- no commentary.',
            '=== PATCH CONSTRAINTS === prefer SMALL, LOCAL changes; modify as few nodes as possible; avoid deleting subtrees or rewriting the root structure.',
            'Allowed: update existing node attributes (bit_start, size_bits, message_type), add/update length bindings (length_of/depends_on), and minimal reparenting only to fix INVALID_PARENT issues.',
            'Discouraged: deleting many nodes, creating many new nodes at once, or changing root topology. Selector topology changes are allowed when needed to merge duplicate/shadow selectors into one shared selector or to restore canonical selector->variant routing.',
            'When size_bits is "variable" but a length field exists, keep size_bits as "variable" and add the length binding instead of guessing a fixed size.',
            'Do NOT introduce new node_type values outside the allowed set.',
        ]

    if normalized_mode == "traffic_fix":
        instructions += [
            "### TRAFFIC ANALYSIS GUIDANCE (CRITICAL):",
            "- Follow the issue priority provided (higher severity + higher sample hits + earlier nodes first).",
            "- If you see 'Parsing blocked' or coverage gaps: address sizing/position issues before tweaking enums/ranges.",
            "- Constraint violations are ambiguous: a violated enum/range can mean the offset, selected variant, message_type, or length binding is wrong. Do NOT remove/widen constraints solely because one traffic sample violates them.",
            "- If a node-level constraint is genuinely wrong, refine it instead of silently deleting it: move it to the correct variant, replace it with a conditional `formula:` constraint, or remove only unsupported constraint syntax that cannot be evaluated.",
            "- If a leaf field has size_bits=\"variable\" (or missing) AND there is no length_of binding to it, it will consume 0 bytes and create coverage gaps. Use DOCUMENTATION CONTEXT to identify the length/count dependency, then add a length_of edge from the controlling field to that leaf (convert units to bits: bytes*8, words/registers*16, etc).",
            "=== TRAFFIC ERROR CONTEXT ===",
            "You will receive structured traffic failures with packet_index, node_id, failure_kind, bit_start, size_bits_eval, total_bits, max_bit_reached, path_node_ids, and context_field_values (values of preceding integer-like fields). Length mismatches also include src/dst/expected/actual bits plus the length_of formula. Some nodes may also include size_bits candidates derived from traffic such as 'val(F) * 8' or '(val(F) - k) * 8'.",
            "=== PATCH CONSTRAINTS (TRAFFIC) ===",
            "1) Prefer SMALL, LOCAL changes; focus on nodes mentioned in traffic failures.",
            "2) Allowed: adjust size_bits or bit_start with simple formulas, refine constraints with documented/traffic-supported formulas, update message_type/data_type when needed, and add/update length_of/condition_on/depends_on edges that control length or presence.",
            "3) Disallowed: large restructures, deleting subtrees, complex arithmetic (no max/min/conditionals), or new selector topologies unless failures + docs clearly require it.",
            "4) When candidates are provided for a node, prefer them or a very small variation.",
            "- If you see candidate-variant failures (oob_seek/oob_read during speculative parsing): FIX those variant size_bits/bit_start issues first. Payload-fill can hide coverage gaps but will not repair broken variants.",
            "- Use traffic_payload_fill only as a LAST RESORT when the remaining bytes are genuinely undocumented/opaque and no length/count binding can be inferred without introducing routing/structure failures.",
            "- traffic_payload_fill MUST be a single new bytes-like field (node_type=field, data_type contains 'bytes' or 'byte').",
            "- traffic_payload_fill MUST be additive-only: use new_nodes only and leave node_updates/new_edges/edge_updates/edge_removes/nodes_to_remove empty (parent linking is automatic via parent_id).",
            "- For traffic_payload_fill formulas: NEVER reference the *parent* node's size_bits (e.g., \"<parent_id>.size_bits\"); this triggers a circular dependency and will be rejected.",
            "- For traffic_payload_fill formulas: do NOT use max()/min(). Keep expressions linear and monotonic.",
            "- Preferred payload fill pattern (use the length_of formula from failures):",
            "    bit_start = \"<last_consumed_leaf>.bit_start + <last_consumed_leaf>.size_bits\"",
            "    size_bits = \"(<dst>.bit_start + (<length_of_formula_bits>)) - (<bit_start>)\"",
        ]

    if atomic_mode:
        # Prepend atomicity constraints to narrow patch size
        instructions = [
            'Atomic mode: perform exactly one focused change cluster. Do not try to solve all pending issues in one patch.',
            'Hard patch-size limit in atomic mode: new_nodes <= 2, node_updates <= 4, new_edges <= 3, edge_updates <= 3, edge_removes <= 3, nodes_to_remove <= 2.',
            'If a repair would require adding many variants or many exception branches, return the first smallest safe branch only, or return {}.',
            'Do not bundle unrelated edits. If multiple fixes are required, return only the first step.',
        ] + instructions

    objective_block = "### CURRENT OBJECTIVE\n" + "\n".join(f"- {line}" for line in objective_lines) + "\n"
    instructions_block = "### INSTRUCTIONS\n" + "\n".join(f"- {text}" for text in instructions) + "\n"

    format_warning_block = ""
    if isinstance(previous_attempt, dict):
        raw_patch = previous_attempt.get("raw_patch")
        if isinstance(raw_patch, dict):
            edge_updates = raw_patch.get("edge_updates") or []
            if any(isinstance(entry, dict) and ("op" in entry or "path" in entry) for entry in edge_updates):
                format_warning_block = (
                    "### PATCH FORMAT WARNINGS\n"
                    "- Previous patch attempted to remove edges using JSON Patch (fields `op`/`path`). This format is unsupported, so no edges were deleted. Use `edge_removes` or `edge_updates` with `edge_identifier` instead.\n"
    )

    issue_examples_block = build_issue_example_block(issues_payload, validator_errors, validator_extras)

    validator_rules = """### VALIDATOR LAWS (NON-NEGOTIABLE)
1.  **The Containment Law**: A parent node's `size_bits` MUST be greater than or equal to the sum of its fixed-size children.
    * Violation: "Children may exceed parent size".
    * Fix: Increase parent size or remove/reparent children.

2.  **The Exclusivity Law**: Two variants cannot overlap in physical space (same `bit_start`) UNLESS they are controlled by the SAME shared selector and have mutually exclusive `condition_on` formulas.
    * Violation: "Invalid overlap with sibling".
    * Fix: Route BOTH variants from one shared selector (e.g., `val(x)==1` vs `val(x)==129`). Duplicating the selector inside each variant does NOT fix the violation.

3.  **The Causality Law (No Future References)**: A node's formula can ONLY reference nodes that appear **before** it in the parsing order.
    * Violation: "Reference to unparsed node".
    * Fix: Remove forward references.

4.  **The Connectivity Law**: If a formula references `val(X)`, there MUST be a corresponding graph edge from node X to the current node.
    * Violation: "References val(X) but no ... edge exists".
    * Fix: Add the missing `length_of` or `condition_on` edge.

5.  **The Type Law**:
    * `length_of` edges determine **size**.
    * `condition_on` edges determine **existence/variant selection**.
    * Do not mix them.

6.  **The "Unknown" Law**: If a container's size depends on variable-sized children (payloads), its size is `variable`. Do NOT force a static formula unless documentation/traffic proves it.
"""
    validator_rules_block = validator_rules + "\n"

    patch_format = (
        "\n{\n"
        "  \"patch_metadata\": {\n"
        "    \"description\": \"...\",\n"
        "    \"intent\": \"...\"\n"
        "  },\n"
        "  \"new_nodes\": [],\n"
        "  \"nodes_to_remove\": [],\n"
        "  \"node_updates\": [\n"
        "    { \"node_id\": 17, \"property\": \"size_bits\", \"value\": \"(val(5)-2)*8\", \"reason\": \"...\" },\n"
        "    { \"node_id\": 9, \"updates\": { \"constraints\": [\"enum: 1|2|3\"] }, \"reason\": \"...\" }\n"
        "  ],\n"
        "  \"new_edges\": [\n"
        "    { \"src\": 5, \"dst\": 17, \"rel\": \"length_of\", \"formula\": \"val(5)*8\" },\n"
        "    { \"src\": 4, \"dst\": 23, \"rel\": \"condition_on\", \"formula\": \"val(4)==3\", \"message_type\": \"request\" }\n"
        "  ],\n"
        "  \"edge_updates\": [\n"
        "    {\n"
        "      \"update_type\": \"modify|remove\",\n"
        "      \"edge_identifier\": { \"src\": \"<SRC_NODE_ID>\", \"dst\": \"<DST_NODE_ID>\", \"rel\": \"length_of\" },\n"
        "      \"new_properties\": {\n"
        "        \"formula\": \"<UPDATED_FORMULA>\"\n"
        "      }\n"
        "    }\n"
        "  ],\n"
        "  \"edge_removes\": [\n"
        "    { \"src\": \"<SRC_NODE_ID>\", \"dst\": \"<DST_NODE_ID>\", \"rel\": \"length_of\" }\n"
        "  ],\n"
        "  \"validation_notes\": []\n"
        "}\n"
    )

    prompt_template = """## PATCH REFINEMENT OBJECTIVE

    ### OUTPUT FORMAT LAWS
    - Return one valid JSON object only, with double-quoted keys and strings.
    - Use ASCII only in JSON strings. Do not emit Unicode math or arrow symbols.
    - Constraint strings must use only these canonical forms: "enum: 1|2|3", "range: 1..4", "formula: val(7) >= 1 and val(7) <= 4", or `constraint_ir: {{"kind":"...","expr":"...","field_ids":[...]}}` for non-executable relation knowledge.
    - If you cannot produce valid JSON under the patch-size limit, return {{}}.

    ### CURRENT TREE
    {tree_json}

    {validator_rules_block}

    {experience_block}

    {objective_block}{pending_block}
    {recent_block}
    {validator_block}
    {followup_block}{feedback_block}
    {avoid_block}
    ### DOCUMENTATION CONTEXT (FOCUS {batch_start}–{batch_end})
    {sections_json}

    {instructions_block}{issue_examples_block}
    ### PATCH SCHEMA (RETURN JSON ONLY)
    {patch_format}
    """

    prompt = prompt_template.format(
        tree_json=tree_json,
        validator_rules_block=validator_rules_block,
        experience_block=experience_block,
        objective_block=objective_block,
        pending_block=pending_block,
        recent_block=recent_block,
        validator_block=validator_block,
        followup_block=followup_block,
        feedback_block=feedback_block + format_warning_block,
        avoid_block=avoid_block,
        batch_start=batch_start,
        batch_end=batch_start + batch_size - 1,
        sections_json=sections_json,
        instructions_block=instructions_block,
        issue_examples_block=("\n" + issue_examples_block if issue_examples_block else ""),
        patch_format=patch_format,
    )

    message = [
        {"role": "system", "content": "You are a protocol tree patch generator. Generate valid ASCII JSON patches only. If no patch is needed, return {}."},
        {"role": "user", "content": prompt},
    ]
    return message
