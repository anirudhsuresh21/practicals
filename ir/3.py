def page_rank(graph, uninvGraph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    num_nodes = len(graph)  
    initial_pr = 1.0
    page_rank = {node: initial_pr for node in graph}  
    out_degrees = {node: len(uninvGraph[node]) for node in uninvGraph}
    for _ in range(max_iterations):
        prev_page_rank = page_rank.copy()
        total_diff = 0.0
        for node in graph:  
            page_rank[node] = (1 - damping_factor)
            node_total = 0
            for neighbor in graph[node]:  
                node_total += page_rank[neighbor] / out_degrees[neighbor]  
            page_rank[node] += damping_factor * node_total
            diff = abs(prev_page_rank[node] - page_rank[node])  
            total_diff += diff  
        if total_diff < tolerance:  
            break
    return page_rank

def invertGraph(graph):
    newGraph = dict()
    for node in graph:
        newGraph[node] = list()
    for node in graph:
        neightbors = graph[node]
        for each in neightbors:
            newGraph[each].append(node)
    return newGraph

graph = {  
    'A': ['B', 'C'],  
    'B': ['C'],  
    'C': ['A']
} 
newGraph = invertGraph(graph)
ranks = page_rank(graph=newGraph, uninvGraph=graph)
print("\n\nCalculating Page Ranks...")
print("\nThe Graph is:")
print(graph)
print("\nThe Page Ranks are:")
print(ranks)
