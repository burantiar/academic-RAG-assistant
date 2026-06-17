"""
Convert a generated researcher profile JSON into a simple graph JSON.

This is a visualization bridge, not the final graph extraction system.

Input:
    outputs/<researcher_id>_profile.json

Output:
    outputs/<researcher_id>_profile_graph.json

The graph can then be loaded by a browser/D3 viewer.
"""

import argparse
import json
import re
from pathlib import Path


OUTPUT_DIR = Path("outputs")


CATEGORY_TO_NODE_TYPE = {
    "research_interests": "ResearchTopic",
    "projects_and_grants": "Project",
    "collaborators": "Person",
    "affiliations": "Institution",
    "publications": "Publication",
    "partnership_opportunities": "Opportunity",
}


CATEGORY_TO_EDGE_TYPE = {
    "research_interests": "works_on",
    "projects_and_grants": "associated_with",
    "collaborators": "collaborates_with",
    "affiliations": "affiliated_with",
    "publications": "authored_or_published",
    "partnership_opportunities": "potential_partner_for",
}


def slugify(value):
    """
    Convert text into a stable-ish graph ID fragment.
    """
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:80]


def short_label(text, max_length=90):
    """
    Make long profile claims readable as graph node labels.
    """
    text = " ".join(text.split())

    if len(text) <= max_length:
        return text

    return text[: max_length - 3].rstrip() + "..."


def get_claim_text(item, category):
    """
    Extract the display claim from a profile item.

    Partnership opportunities use `opportunity`; other categories use `claim`.
    """
    if category == "partnership_opportunities":
        return item.get("opportunity", "")

    return item.get("claim", "")


def build_graph(profile):
    """
    Convert profile categories into nodes and edges.
    """
    researcher_id = profile.get("researcher_id", "researcher")
    researcher_label = researcher_id.replace("_", " ").title()
    person_id = f"person:{slugify(researcher_id)}"

    nodes = [
        {
            "id": person_id,
            "type": "Person",
            "name": researcher_label,
        }
    ]

    edges = []
    seen_nodes = {person_id}

    for category, node_type in CATEGORY_TO_NODE_TYPE.items():
        items = profile.get(category, [])

        if not isinstance(items, list):
            continue

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            claim_text = get_claim_text(item, category)

            if not claim_text:
                continue

            node_id = f"{node_type.lower()}:{slugify(claim_text)}"

            if node_id not in seen_nodes:
                nodes.append(
                    {
                        "id": node_id,
                        "type": node_type,
                        "name": short_label(claim_text),
                        "full_claim": claim_text,
                    }
                )
                seen_nodes.add(node_id)

            edges.append(
                {
                    "source": person_id,
                    "target": node_id,
                    "type": CATEGORY_TO_EDGE_TYPE[category],
                    "confidence": item.get("confidence"),
                    "citations": item.get("citations")
                    or item.get("supporting_citations")
                    or [],
                    "rationale": item.get("rationale"),
                }
            )

    return {
        "nodes": nodes,
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert a generated researcher profile into graph JSON."
    )
    parser.add_argument(
        "researcher_id",
        help="Researcher ID, e.g. natalia_efremova",
    )

    args = parser.parse_args()

    profile_path = OUTPUT_DIR / f"{args.researcher_id}_profile.json"
    graph_path = OUTPUT_DIR / f"{args.researcher_id}_profile_graph.json"

    with profile_path.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    graph = build_graph(profile)

    with graph_path.open("w", encoding="utf-8") as file:
        json.dump(graph, file, indent=2, ensure_ascii=False)

    print(f"Wrote graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    print(f"Output: {graph_path}")


if __name__ == "__main__":
    main()