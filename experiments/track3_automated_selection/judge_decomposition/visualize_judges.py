#!/usr/bin/env python3
"""Interactive visualization of judge decomposition hierarchies.

Creates an interactive HTML graph showing the parent-child relationships
between judges in a decomposition YAML file.

Usage:
    python experiments/track3_automated_selection/visualize_judges.py \
        experiments/track3_automated_selection/generated_judges/test-depth2-truthfulness-*.yaml

    # Or open in browser automatically:
    python experiments/track3_automated_selection/visualize_judges.py \
        path/to/judges.yaml --open
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

try:
    from pyvis.network import Network
except ImportError:
    print("pyvis not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis"])
    from pyvis.network import Network


# Color palette for different depths
DEPTH_COLORS = [
    "#e63946",  # Red - root
    "#f4a261",  # Orange - depth 1
    "#2a9d8f",  # Teal - depth 2
    "#264653",  # Dark blue - depth 3
    "#9b5de5",  # Purple - depth 4
    "#00bbf9",  # Light blue - depth 5+
]


def load_judges_yaml(yaml_path: Path) -> List[Dict[str, Any]]:
    """Load judges from a YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if isinstance(data, dict) and "judges" in data:
        return data["judges"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected YAML structure in {yaml_path}")


def build_judge_graph(judges: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup dictionary and compute relationships."""
    graph: Dict[str, Dict[str, Any]] = {}
    
    for judge in judges:
        judge_id = judge.get("id", judge.get("judge_id", "unknown"))
        parent_id = judge.get("parent_id")
        
        graph[judge_id] = {
            "id": judge_id,
            "name": judge.get("name", judge_id),
            "description": judge.get("description", ""),
            "parent_id": parent_id,
            "children": [],
            "depth": 0,
        }
    
    # Build children lists
    for judge_id, node in graph.items():
        parent_id = node["parent_id"]
        if parent_id and parent_id in graph:
            graph[parent_id]["children"].append(judge_id)
    
    # Compute depths (BFS from roots)
    roots = [jid for jid, node in graph.items() if not node["parent_id"] or node["parent_id"] not in graph]
    
    def set_depth(node_id: str, depth: int) -> None:
        graph[node_id]["depth"] = depth
        for child_id in graph[node_id]["children"]:
            set_depth(child_id, depth + 1)
    
    for root_id in roots:
        set_depth(root_id, 0)
    
    return graph


def create_interactive_graph(
    graph: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Judge Decomposition Hierarchy",
    height: str = "800px",
    width: str = "100%",
) -> Path:
    """Create an interactive Pyvis network visualization."""
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#222222",
        font_color="white",
        heading=title,
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "arial"
            },
            "scaling": {
                "min": 20,
                "max": 50
            }
        },
        "edges": {
            "color": {
                "inherit": true
            },
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "vertical"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 150,
                "springConstant": 0.01,
                "nodeDistance": 200
            },
            "solver": "hierarchicalRepulsion"
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 200,
                "nodeSpacing": 150
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": {
                "enabled": true
            }
        }
    }
    """)
    
    # Add nodes
    for judge_id, node in graph.items():
        depth = node["depth"]
        color = DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)]
        
        # Create tooltip with full info
        tooltip = f"""<b>{node['name']}</b><br>
ID: {judge_id}<br>
Depth: {depth}<br>
Children: {len(node['children'])}<br><br>
{node['description'][:200]}{'...' if len(node['description']) > 200 else ''}"""
        
        # Size based on number of children (larger = more children)
        size = 25 + len(node["children"]) * 5
        
        # Label: use short name
        label = node["name"]
        if len(label) > 25:
            label = label[:22] + "..."
        
        net.add_node(
            judge_id,
            label=label,
            title=tooltip,
            color=color,
            size=size,
            shape="dot" if node["children"] else "diamond",
            borderWidth=2,
            borderWidthSelected=4,
        )
    
    # Add edges
    for judge_id, node in graph.items():
        parent_id = node["parent_id"]
        if parent_id and parent_id in graph:
            net.add_edge(parent_id, judge_id, width=2)
    
    # Save to HTML
    net.save_graph(str(output_path))
    
    # Inject custom CSS for better styling
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    custom_css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
        }
        h1 {
            text-align: center;
            color: #ffffff;
            padding: 20px;
            margin: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        #mynetwork {
            border: 1px solid #444;
        }
        .legend {
            position: fixed;
            top: 80px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            z-index: 1000;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .stats {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            z-index: 1000;
        }
    </style>
    """
    
    # Count stats
    depths = [node["depth"] for node in graph.values()]
    max_depth = max(depths) if depths else 0
    root_count = sum(1 for node in graph.values() if not node["parent_id"] or node["parent_id"] not in graph)
    leaf_count = sum(1 for node in graph.values() if not node["children"])
    
    legend_html = f"""
    <div class="legend">
        <b>Legend (Depth)</b>
        <div class="legend-item"><div class="legend-color" style="background:{DEPTH_COLORS[0]}"></div>Root (0)</div>
        <div class="legend-item"><div class="legend-color" style="background:{DEPTH_COLORS[1]}"></div>Depth 1</div>
        <div class="legend-item"><div class="legend-color" style="background:{DEPTH_COLORS[2]}"></div>Depth 2</div>
        <div class="legend-item"><div class="legend-color" style="background:{DEPTH_COLORS[3]}"></div>Depth 3+</div>
        <hr style="border-color:#555">
        <div>● Parent node</div>
        <div>◆ Leaf node</div>
    </div>
    <div class="stats">
        <b>Statistics</b><br>
        Total judges: {len(graph)}<br>
        Root judges: {root_count}<br>
        Leaf judges: {leaf_count}<br>
        Max depth: {max_depth}
    </div>
    """
    
    # Insert custom elements
    html = html.replace("</head>", f"{custom_css}</head>")
    html = html.replace("</body>", f"{legend_html}</body>")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_path


def print_tree(graph: Dict[str, Dict[str, Any]], node_id: str, prefix: str = "", is_last: bool = True) -> None:
    """Print a text tree representation."""
    node = graph[node_id]
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{node['name']}")
    
    children = node["children"]
    for i, child_id in enumerate(children):
        new_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(graph, child_id, new_prefix, i == len(children) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize judge decomposition hierarchy as an interactive graph"
    )
    parser.add_argument(
        "yaml_file",
        type=Path,
        help="Path to the judges YAML file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output HTML file path (default: same name as input with .html extension)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the visualization in default browser after creation",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Also print a text tree representation to console",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the visualization",
    )
    
    args = parser.parse_args()
    
    if not args.yaml_file.exists():
        print(f"Error: File not found: {args.yaml_file}")
        return
    
    print(f"Loading judges from: {args.yaml_file}")
    judges = load_judges_yaml(args.yaml_file)
    print(f"Found {len(judges)} judges")
    
    graph = build_judge_graph(judges)
    
    # Print text tree if requested
    if args.tree:
        print("\n" + "=" * 60)
        print("Judge Hierarchy Tree")
        print("=" * 60)
        roots = [jid for jid, node in graph.items() if not node["parent_id"] or node["parent_id"] not in graph]
        for i, root_id in enumerate(roots):
            if i > 0:
                print()
            print_tree(graph, root_id, "", True)
        print("=" * 60 + "\n")
    
    # Determine output path
    output_path = args.output or args.yaml_file.with_suffix(".html")
    
    # Create title
    title = args.title or f"Judge Decomposition: {args.yaml_file.stem}"
    
    print(f"Creating interactive visualization...")
    create_interactive_graph(graph, output_path, title=title)
    print(f"✓ Visualization saved to: {output_path}")
    
    if args.open:
        print("Opening in browser...")
        webbrowser.open(f"file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
