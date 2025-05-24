import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import time
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def visualize_mcts_tree(root, max_depth=2):
    """
    Visualize the MCTS tree starting from the root node, up to max_depth levels
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"MCTS Tree (depth {max_depth})")
    
    # Draw the tree
    _draw_tree(ax, root, max_depth)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def _draw_tree(ax, node, max_depth, x=0, y=0, level=0, horizontal_spacing=2.0):
    """Recursively draw the tree"""
    if level > max_depth:
        return
    
    # Draw the current node
    node_color = 'lightblue' if node.is_maximizing else 'lightgreen'
    if node.is_terminal:
        node_color = 'lightcoral'
    
    width = 1.8
    height = 0.8
    
    rect = Rectangle((x - width/2, y - height/2), width, height, 
                    facecolor=node_color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect)
    
    # Add node text
    if node.move:
        move_text = f"{node.move}"
    else:
        move_text = "root"
    
    node_text = f"{move_text}\nV: {node.visit_count}\nQ: {node.value:.3f}"
    ax.text(x, y, node_text, ha='center', va='center', fontsize=8)
    
    # Draw children
    if not node.children or level == max_depth:
        return
    
    children = list(node.children.items())
    if len(children) > 8:  # Limit number of displayed children for clarity
        children = sorted(children, key=lambda x: x[1].visit_count, reverse=True)[:8]
    
    n_children = len(children)
    total_width = horizontal_spacing * (n_children - 1)
    
    for i, (move, child) in enumerate(children):
        child_x = x - total_width/2 + i * horizontal_spacing
        child_y = y - 1.5
        
        # Draw line to child
        ax.plot([x, child_x], [y - height/2, child_y + height/2], 'k-', alpha=0.5)
        
        # Recursively draw the child
        _draw_tree(ax, child, max_depth, child_x, child_y, level+1, horizontal_spacing/1.5)

def analyze_mcts_results(root_node):
    """
    Analyze the results of an MCTS search and print useful statistics
    """
    print("\nMCTS Search Analysis:")
    print("=====================")
    
    # Calculate total visits
    total_visits = sum(child.visit_count for child in root_node.children.values())
    
    # Sort children by visit count
    sorted_children = sorted(
        root_node.children.items(), 
        key=lambda x: x[1].visit_count, 
        reverse=True
    )
    
    print(f"{'Move':<15} {'Visits':<10} {'%':<8} {'Value':<10} {'Prior':<10}")
    print("-" * 55)
    
    for move, node in sorted_children:
        visit_pct = 100 * node.visit_count / total_visits if total_visits > 0 else 0
        print(f"{str(move):<15} {node.visit_count:<10d} {visit_pct:<8.2f} {node.value:<10.4f} {node.prior_prob:<10.4f}")


def run_mcts_benchmark(player, board, num_runs=3):
    """Run a benchmark to measure MCTS performance with different parameters"""
    print("\nMCTS Performance Benchmark:")
    print("=========================")
    
    # Test different numbers of simulations
    simulation_counts = [100, 400, 800, 1600]
    
    for sim_count in simulation_counts:
        player.num_simulations = sim_count
        
        total_time = 0
        nodes_visited = 0
        
        for _ in range(num_runs):
            board_copy = deepcopy(board)
            start_time = time.time()
            move, stats = player.get_move_with_stats(board_copy)
            end_time = time.time()
            
            run_time = end_time - start_time
            total_time += run_time
            nodes_visited += sum(stat["visits"] for stat in stats.values())
        
        avg_time = total_time / num_runs
        avg_nodes = nodes_visited / num_runs
        nodes_per_second = avg_nodes / avg_time
        
        print(f"Simulations: {sim_count}, Time: {avg_time:.3f}s, Nodes: {avg_nodes:.1f}, Speed: {nodes_per_second:.1f} nodes/s")
    
    # Reset to original settings
    player.num_simulations = 800

def benchmark_mcts(player, board, simulation_counts=[100, 400, 800]):
    """
    Benchmark the MCTS algorithm with different simulation counts
    """
    original_count = player.num_simulations
    
    print("\nMCTS Benchmark:")
    print("==============")
    print(f"{'Simulations':<15} {'Time (s)':<10} {'Nodes/s':<10}")
    print("-" * 40)
    
    for sim_count in simulation_counts:
        player.num_simulations = sim_count
        
        total_time = 0
        total_nodes = 0
        runs = 3
        
        for _ in range(runs):
            board_copy = deepcopy(board)
            start_time = time.time()
            
            # Create root node 
            root = player._create_root_node(board_copy)
            
            # Run MCTS for specified number of simulations
            simulation_count = 0
            while simulation_count < sim_count:
                leaf, board_state = player._select_and_expand(root, board_copy)
                value = player._evaluate_leaf(leaf, board_state)
                player._backpropagate(leaf, value)
                simulation_count += 1
            
            end_time = time.time()
            total_time += end_time - start_time
            total_nodes += sim_count
        
        avg_time = total_time / runs
        nodes_per_sec = total_nodes / total_time if total_time > 0 else 0
        
        print(f"{sim_count:<15d} {avg_time:<10.3f} {nodes_per_sec:<10.1f}")
    
    # Restore original simulation count
    player.num_simulations = original_count


def top_k_filtering_test(player, board, k_values=[3, 5, 10, None]):
    """
    Test the effect of top-k move filtering on MCTS performance
    """
    original_k = player.top_k_moves
    original_sims = player.num_simulations
    player.num_simulations = 400  # Fixed number of simulations for testing
    
    print("\nTop-K Filtering Test:")
    print("===================")
    print(f"{'Top-K':<10} {'Time (s)':<10} {'Best Move':<15} {'Confidence':<10}")
    print("-" * 50)
    
    for k in k_values:
        player.top_k_moves = k if k is not None else 0
        k_str = str(k) if k is not None else "None"
        
        board_copy = deepcopy(board)
        start_time = time.time()
        
        # Create root node with top-k filtering
        root = player._create_root_node(board_copy)
        
        # Run MCTS
        simulation_count = 0
        while simulation_count < player.num_simulations:
            leaf, board_state = player._select_and_expand(root, board_copy)
            value = player._evaluate_leaf(leaf, board_state)
            player._backpropagate(leaf, value)
            simulation_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get the best move and its confidence
        best_child = max(root.children.values(), key=lambda node: node.visit_count)
        best_move = best_child.move
        
        # Calculate confidence (% of visits to best move)
        total_visits = sum(child.visit_count for child in root.children.values())
        confidence = 100 * best_child.visit_count / total_visits if total_visits > 0 else 0
        
        print(f"{k_str:<10} {total_time:<10.3f} {str(best_move):<15} {confidence:<10.2f}%")
    
    # Restore original parameters
    player.top_k_moves = original_k
    player.num_simulations = original_sims
