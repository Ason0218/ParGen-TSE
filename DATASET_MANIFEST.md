# Dataset Manifest

## Protocol Inputs

The dataset contains inputs for eight protocols:

| Protocol | Directory |
| --- | --- |
| Modbus | `dataset/pargen/data/modbus/` |
| BFD | `dataset/pargen/data/bfd/` |
| TCP | `dataset/pargen/data/tcp/` |
| DHCP | `dataset/pargen/data/dhcp/` |
| IPv4 | `dataset/pargen/data/ipv4/` |
| ICMPv4 | `dataset/pargen/data/icmp/` |
| DNS | `dataset/pargen/data/dns/` |
| PFCP | `dataset/pargen/data/pfcp/` |

Each protocol directory holds these inputs (under `inputs/` unless noted):

| Path | Content |
| --- | --- |
| `pdfs/` | The official specification documents (PDF, RFC text, or the 3GPP Word document for PFCP). |
| `pcap/`, `traffic/` | The evaluation trace and its payload dump. |
| `traffic/refinement_packets.txt` (Modbus) | Four unlabeled messages used by the documented Modbus refinement example. |
| `gt/` | Packet-level expectations used for scoring; Modbus instead uses `<protocol>/gt/` outside `inputs/`. |
| `stage_c_codegen_profile*.json` | Platform registration for each target (dissector table, port, filter name). |

## Protocol-Knowledge Ground Truth

| Directory | Content |
| --- | --- |
| `dataset/pargen/protocol_knowledge_gt/` | Field and constraint-relation annotations for the paper protocols. |
