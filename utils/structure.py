from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS
from collections import defaultdict

# Define a function to recursively build the class hierarchy
def build_class_tree(graph, current_class, tree):
    # Find all subclasses of the current class
    subclasses = graph.subjects(predicate=RDFS.subClassOf, object=current_class)
    for subclass in subclasses:
        tree[str(current_class).split('#')[-1]].append(str(subclass).split('#')[-1])
        # Recursively build the tree for each subclass
        build_class_tree(graph, subclass, tree)

# Start building the tree from the Point root
def get_tree():
    # Load the TTL file
    file_path = 'utils/Brick.ttl'
    g = Graph()
    g.parse(file_path, format='turtle')

    root_class = None
    for s, p, o in g:
        if str(s).endswith("Point") and p == RDF.type and str(o).endswith("Class"):
            root_class = s
            break

    if not root_class:
        raise ValueError("Point root class not found in the TTL file.")

    # Create a dictionary to store the tree structure
    tree_structure = defaultdict(list)

    # Build the tree
    build_class_tree(g, root_class, tree_structure)
    return tree_structure