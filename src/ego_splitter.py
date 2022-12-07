"""Ego-Splitter class"""

# Ignore these.. kpjim wants to use his custom made "community" package
import sys
import os
t = os.getcwd() + "/python-louvain"
sys.path.insert(0, t)

import community
import networkx as nx
from tqdm import tqdm
import time


class EgoNetSplitter(object):
    """An implementation of `"Ego-Splitting" see:
    https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf
    From the KDD '17 paper "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters".
    The tool first creates the egonets of nodes.
    A persona-graph is created which is clustered by the Louvain method.
    The resulting overlapping cluster memberships are stored as a dictionary.
    Args:
        resolution (float): Resolution parameter of Python Louvain. Default 1.0.
    """
    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def _create_egonet(self, node, update = 0):
        """
        Creating an ego net, extracting personas and partitioning it.

        Args:
            node: Node ID for egonet (ego node).
        """
        if update != 0:
            # kpjim: Remove old personalities of node from the persona
            if node in self.personalities:
                for p in self.personalities[node]:
                    self.persona_graph.remove_node(p)
        ego_net_minus_ego = self.graph.subgraph(self.graph.neighbors(node))
        components = {i: n for i, n in enumerate(nx.connected_components(ego_net_minus_ego))}
        new_mapping = {}
        personalities = []
        for k, v in components.items():
            personalities.append(self.index)
            if update != 0:
                # kpjim: Update now the personality_map so that we don't have to
                # iterate it later
                self.personality_map[self.index] = node
            for other_node in v:
                new_mapping[other_node] = self.index
            self.index = self.index+1
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def _create_egonets(self, R_ego_t = None, update = 0):
        """
        Creating an egonet for each node. kpjim: Use R_ego_t to restrict the
        egonet recalculations
        """
        """ kpjim:
        if R_ego_t is not None we have already calculated components,
        personalities and indexing once and now we are in the process of
        updating using a batch. So we don't need (actually we NEED NOT TO)
        initialize them again!
        """
        if R_ego_t == None:
            self.components = {}
            self.personalities = {}
            self.index = 0
        print("Creating egonets.")
        nodes = R_ego_t if R_ego_t != None else self.graph.nodes()
        for node in tqdm(nodes):
            self._create_egonet(node, update)

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {p: n for n in self.graph.nodes() for p in self.personalities[n]}

    def _get_new_edge_ids(self, edge):
        """
        Getting the new edge identifiers.
        Args:
            edge: Edge being mapped to the new identifiers.
        """
        return (self.components[edge[0]][edge[1]], self.components[edge[1]][edge[0]])

    def _create_persona_graph(self):
        """
        Create a persona graph using the egonet components.
        """
        print("Creating the persona graph.")
        self.persona_graph_edges = [self._get_new_edge_ids(e) for e in tqdm(self.graph.edges())]
        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)
    
    def _update_persona_graph(self, Rt):
        """
        kpjim: Update the _already created_ persona graph by visiting only the
        nodes in the Rt set
        """
        print("Updating the persona graph.")
        for n in Rt:
            for e in self.graph.edges(n):
                x, y = self._get_new_edge_ids(e)
                self.persona_graph.add_edge(x, y)

    def _create_partitions(self):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        print("Clustering the persona graph.")
        self.partitions = community.best_partition(self.persona_graph, resolution=self.resolution)
        self.overlapping_partitions = {node: [] for node in self.graph.nodes()}
        self.final_partitions = dict()
        for node, membership in self.partitions.items():
            self.overlapping_partitions[self.personality_map[node]].append(membership)
            if membership not in self.final_partitions:
                self.final_partitions[membership] = set()
            self.final_partitions[membership].add(self.personality_map[node])

    def fit(self, graph):
        """
        Fitting an Ego-Splitter clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        start = time.time()
        self._create_egonets()
        self._map_personalities()
        self._create_persona_graph()
        end = time.time()
        self._create_partitions()
        return end - start

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.
        Return types:
            * **memberships** *(dictionary of lists)* - Cluster memberships.
        """
        return self.overlapping_partitions

    def get_overlaps(self):
        return self.final_partitions

    def _update(self, R_ego_t):
        start = time.time()
        self._create_egonets(R_ego_t = R_ego_t, update = 1)
        # We have already updated the personality map
        ## self._map_personalities()
        ## self._create_persona_graph()
        self._update_persona_graph(R_ego_t)
        end = time.time()
        self._create_partitions()
        return end - start

    def add_batch(self, batch):
        """Add a new batch of edges to the original graph. And re-fit"""
        R_ego = set()
        for i, j in batch:
            self.graph.add_edge(i, j)
        for i, j in batch:
            R_ego.add(i)
            R_ego.add(j)
            # TODO: Use intersection of i's and j's egonets instead of neighbors
            # here..
            common = set(self.graph.neighbors(i)).intersection(self.graph.neighbors(j))
            for w in common:
                R_ego.add(w)
                #... 
        #self.fit(R_ego_t = R_ego, index = self.index)
        return self._update(R_ego_t = R_ego)

    def stream_insert_edge(self, edge):
        """ Insert edge (x, y) to the original graph and re-fit the graph """
        i, j = edge[0], edge[1]
        self.graph.add_edge(i, j)
        R_ego = set()
        R_ego.add(i)
        R_ego.add(j)
        # TODO: Use intersection of i's and j's egonets instead of neighbors
        # here..
        common = set(self.graph.neighbors(i)).intersection(self.graph.neighbors(j))
        for w in common:
            R_ego.add(w)
        self._update(R_ego)

    def stream_add_batch(self, batch):
        for edge in batch:
            self.stream_insert_edge(edge)

    def kpjim_ego(self):
        print("Hello kpjim's ego")
