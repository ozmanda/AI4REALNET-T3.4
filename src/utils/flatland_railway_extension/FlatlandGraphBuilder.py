import copy
from typing import Dict, Tuple, List
from typing import Union

import networkx as nx
import numpy as np
# import all flatland dependance
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_position_equal, fast_argmax, fast_count_nonzero
from flatland.envs.rail_env_action import RailEnvActions
from matplotlib import pyplot as plt
from networkx.classes.reportviews import OutEdgeView

from src.utils.flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from src.utils.flatland_railway_extension.environments.InfrastructureData import InfrastructureData


class FlatlandGraphBuilder:
    def __init__(self,
                 railroad_switch_analyser: RailroadSwitchAnalyser,
                 infrastructure_data: Union[InfrastructureData, None] = None,
                 activate_simplified: bool = False,
                 keep_switch_neighbors_at_simplification: bool = True):
        self.railroad_switch_analyser = railroad_switch_analyser
        self._graph: Union[nx.DiGraph, None] = None
        self._nodes: Union[Dict[str, Tuple[int, int, float]], None] = None
        self.set_infrastructure_data(infrastructure_data)
        self.keep_switch_neighbors_at_simplification = keep_switch_neighbors_at_simplification
        if activate_simplified:
            self.activate_simplified()
        else:
            self.activate_full_graph()

    def set_infrastructure_data(self, infrastructure_data: Union[InfrastructureData, None]):
        self._infrastructure_data = infrastructure_data

    def activate_full_graph(self):
        self._graph, self._nodes, self._from_vertex_edge_map = self._create_full_graph()

    def activate_simplified(self):
        self._graph, self._nodes, self._from_vertex_edge_map = self._create_simplified_graph()

    def get_graph(self) -> Union[nx.DiGraph, None]:
        return self._graph

    def get_nodes(self) -> Union[Dict[str, Tuple[int, int, float]], None]:
        return self._nodes

    def get_edges(self) -> OutEdgeView:
        return self._graph.edges

    def estimate_edge_len(self, pos: Tuple[int, int]) -> float:
        """
        Estimate the cell distance ending in the distance map. The currently implemented heuristic is very simple:
        If the infrastructure data is missing, the cell length becomes one - otherwise the cell length is approximated
        with the physical distance (in meters) and the maximum allowed speed (in meters per second). So the edge length
        is the expected time spent on the item assuming the agent is traveling with maximal allowed speed.
        """
        if self._infrastructure_data is not None:
            edge_len = self._infrastructure_data.get_cell_length(pos)
            edge_vel = self._infrastructure_data.get_velocity(pos)
            return edge_len / edge_vel
        return 1.0

    def _create_full_graph(self):
        graph = nx.DiGraph()
        env = self.railroad_switch_analyser.get_rail_env()
        nodes = {}
        from_vertex_edge_map = {}
        for h in range(env.height):
            for w in range(env.width):
                from_position = (h, w)
                for from_direction in range(4):
                    possible_transitions = env.rail.get_transitions(*from_position, from_direction)
                    nbr_possible_transitions = fast_count_nonzero(possible_transitions)
                    if nbr_possible_transitions > 0:
                        actions = {}
                        for to_direction in range(4):
                            actions.update({to_direction: RailEnvActions.MOVE_FORWARD})
                        if nbr_possible_transitions > 1:
                            action = 0
                            for to_direction in [(from_direction + i) % 4 for i in range(-1, 2)]:
                                if possible_transitions[to_direction]:
                                    actions.update({to_direction: list(RailEnvActions)[action + 1]})
                                action += 1

                        for to_direction in range(4):
                            if possible_transitions[to_direction] == 1:
                                new_position = get_new_position(from_position, to_direction)
                                from_vertex_name = '{}_{}_{}'.format(from_position[0], from_position[1], from_direction)
                                to_vertex_name = '{}_{}_{}'.format(new_position[0], new_position[1], to_direction)
                                graph.add_edge(from_vertex_name,
                                               to_vertex_name,
                                               length=self.estimate_edge_len(from_position),
                                               from_nodes=[from_vertex_name],
                                               resources=[from_position],
                                               action=[actions.get(to_direction)],
                                               resource_id='{}_{}'.format(from_position[0], from_position[1])
                                               )
                                nodes.update({from_vertex_name: (from_position[0], from_position[1], from_direction)})
                                nodes.update({to_vertex_name: (new_position[0], new_position[1], to_direction)})
                                from_vertex_edge_map.update({from_vertex_name: (from_vertex_name, to_vertex_name)})

        return graph, nodes, from_vertex_edge_map

    def _create_simplified_graph(self):
        self._graph, self._nodes, self._from_vertex_edge_map = self._create_full_graph()
        graph, nodes, from_vertex_edge_map = self._graph, self._nodes, self._from_vertex_edge_map
        # loop as long as the graph changes (gets updated)
        graph_updated = True
        while graph_updated:
            graph_updated = False

            for node in list(graph.nodes()):
                #
                # (Full) graph as input:
                #
                # \ .------.      .------.     .------.     .------. /
                #   | Node |----->| Node |---->| Node |---->| Node |
                # / `------`      `------`     `------`     `------` \
                #
                # The inner nodes with exact one in-coming-edge and one out-going edge get removed
                # Simplified version as output:
                #
                # \ .------.                                .------. /
                #   | Node |------------------------------->| Node |
                # / `------`                                `------` \
                #
                if (graph.in_degree(node) == 1 and graph.out_degree(node) == 1):
                    in_dat = None
                    out_dat = None
                    in_vert = None
                    out_vert = None

                    # determine in_data, in_vert and out_dat, out_vert
                    for incoming_u, incoming_v in graph.in_edges(node):
                        # The for loop is not a true loop because only one element is returned.
                        in_dat = graph.get_edge_data(incoming_u, incoming_v)
                        in_vert = incoming_u if node == incoming_v else incoming_v
                    for outgoing_u, outgoing_v in graph.out_edges(node):
                        # The for loop is not a true loop because only one element is returned.
                        out_dat = graph.get_edge_data(outgoing_u, outgoing_v)
                        out_vert = outgoing_v if node == outgoing_u else outgoing_u

                    #
                    # check neighbour nodes (otherwise the methods removes to much nodes)
                    #
                    node_pos, node_dir = self.get_node_pos_dir(node)
                    if self.railroad_switch_analyser.is_diamond_crossing(node_pos):
                        continue
                    if self.railroad_switch_analyser.is_dead_end(node_pos):
                        continue
                    if self.keep_switch_neighbors_at_simplification:
                        if self.railroad_switch_analyser.is_switch_neighbor(node_pos):
                            continue

                    if graph.out_degree(in_vert) == 1 and graph.in_degree(out_vert) == 1:
                        # add new edge
                        graph.add_edge(in_vert, out_vert)

                        # copy edge data by looping over keys
                        data = graph.get_edge_data(in_vert, out_vert)
                        for key in in_dat.keys():
                            data.update({key: copy.copy(in_dat.get(key))})

                        # manually update the resource list
                        resources = in_dat.get('resources')
                        for d in out_dat.get('resources'):
                            resources.append(d)
                        data.update({'resources': copy.copy(resources)})

                        action = in_dat.get('action')
                        data.update({'action´': action})

                        edge_len = len(resources)
                        if self._infrastructure_data is not None:
                            edge_len = 0
                            for r in resources:
                                edge_len += self._infrastructure_data.get_cell_length(r)
                        data.update({'length': edge_len})

                        # manually update the from-nodes (nodes information)
                        from_nodes = in_dat.get('from_nodes')
                        for n in out_dat.get('from_nodes'):
                            from_nodes.append(n)
                        data.update({'from_nodes': copy.copy(from_nodes)})

                        # update the lookup tables
                        for n in copy.copy(from_nodes):
                            from_vertex_edge_map.update({n: (in_vert, out_vert)})

                        # remove "old" edge
                        graph.remove_node(node)

                        # mark that the graph has changes (updated)
                        graph_updated = True
                        break

        self._ordering_edge_resources(graph)

        return graph, nodes, from_vertex_edge_map

    def get_edge_resource(self, edge) -> List[Tuple[int, int]]:
        edge_data = self._graph.get_edge_data(edge[0], edge[1])
        return edge_data.get('resources')

    def _ordering_edge_resources(self, graph: nx.DiGraph):
        # ---------------- ordering edge resource
        for edge in graph.edges:
            edge_data = graph.get_edge_data(edge[0], edge[1])
            resources = edge_data.get('resources')
            if len(resources) > 1:
                pos, direction = self.get_coordinate_direction_from_node_id(edge[0])
                to_pos, _ = self.get_coordinate_direction_from_node_id(edge[1])
                res = [pos]
                env = self.railroad_switch_analyser.get_rail_env()
                while not fast_position_equal(pos, to_pos):
                    possible_transitions = env.rail.get_transitions(*pos, direction)
                    direction = fast_argmax(possible_transitions)
                    pos = get_new_position(pos, direction)
                    if not fast_position_equal(pos, to_pos):
                        res.append(pos)
                edge_data.update({'resources': res})

    def get_edge_weight(self, edge) -> float:
        return float(self.get_graph().get_edge_data(edge[0], edge[1])['length'])

    def make_node_label(self, node) -> str:
        return node

    def make_edge_label(self, edge) -> str:
        res_info = self.get_graph().get_edge_data(edge[0], edge[1])['resource_id']
        label = 'cell_pos:{}|len:{:3.0f}'.format(res_info, self.get_edge_weight(edge))
        return label

    def get_node_pos_dir(self, node_str: str):
        pos_xy_dir = self.get_nodes().get(node_str)
        pos = pos_xy_dir[:2]
        dir = pos_xy_dir[2]
        return pos, dir

    def render(self, render_direction_layer=True):
        nodes = self.get_graph().nodes()
        positions = {}
        for n in nodes:
            pos, dir = self.get_node_pos_dir(n)
            if render_direction_layer:
                x = pos[0] + self.railroad_switch_analyser.get_rail_env().width * (dir % 2)
                y = pos[1] + self.railroad_switch_analyser.get_rail_env().height * (dir // 2)
            else:
                x = pos[0]
                y = pos[1]
            positions.update({n: (y, -x)})

        plt.figure()
        nx.draw(
            self.get_graph(), positions, edge_color='black', width=1, linewidths=1,
            node_size=200, node_color='lightgray', alpha=1.0, font_size=8,
            labels={node: self.make_node_label(node) for node in self.get_graph().nodes()}
        )

        nx.draw_networkx_edge_labels(
            self.get_graph(), positions,
            edge_labels={e: self.make_edge_label(e) for e in self.get_graph().edges()},
            font_color='red', font_size=8
        )
        plt.axis('off')
        plt.show()

    @staticmethod
    def get_coordinate_direction_from_node_id(node_name: str):
        xy_d = node_name.split('_')
        xy = (int(xy_d[0]), int(xy_d[1]))
        direction = int(xy_d[2])
        return xy, direction

    @staticmethod
    def get_resource_id_from_node_id(node_name: str):
        xy_d = node_name.split('_')
        resource_id = '{}_{}'.format(int(xy_d[0]), int(xy_d[1]))
        return resource_id

    def get_mapped_vertex(self, flatland_position, flatland_direction):
        node_key = '{}_{}_{}'.format(flatland_position[0], flatland_position[1], flatland_direction)
        return self._from_vertex_edge_map.get(node_key)

    def get_shortest_path(self, start_position, start_direction, target_position, weight=None):
        '''
        This methods traverse the _graph to get shortest path. The target gets scaned for all four incoming directions.
        :param start_position: 2d coordinate
        :param start_direction: orientation passed a agent moving direction
        :param target_position: 2d coordinate of target position
        :return: shortest path, all paths, len of all found paths
        '''
        from_edge = self.get_mapped_vertex(start_position, start_direction)
        paths = []
        paths_len = []
        for target_direction in range(4):
            path = []
            to_edge = self.get_mapped_vertex(target_position, target_direction)
            if from_edge is not None and to_edge is not None:
                from_node_intern = from_edge[0]
                to_node_intern = to_edge[1]
                fn = self.get_graph().nodes.get(from_node_intern, None)
                tn = self.get_graph().nodes.get(to_node_intern, None)

                if fn is not None and tn is not None:
                    try:
                        s_path = nx.shortest_path(self.get_graph(), from_node_intern, to_node_intern, weight=weight)
                        start_p = from_node_intern
                        for p in s_path:
                            if start_p != p:
                                # xy, _ = FlatlandGraphBuilder.get_coordinate_direction_from_node_id(p)
                                resources = self.get_graph().get_edge_data(start_p, p).get('resources')
                                append_ok = start_position not in resources
                                for res in resources:
                                    if fast_position_equal(start_position, res):
                                        append_ok = True
                                    if append_ok:
                                        path.append(res)
                                    if fast_position_equal(target_position, res):
                                        append_ok = False
                            start_p = p
                    except nx.NetworkXNoPath:
                        pass
            paths.append(path)
            path_len = len(path)
            if path_len == 0:
                path_len = np.inf
            paths_len.append(path_len)
        arg_sorted_path_len = np.argsort(paths_len)
        return paths[arg_sorted_path_len[0]], paths, paths_len

    def get_shortest_path_edges(self, start_position, start_direction, target_position, weight=None):
        '''
        This methods traverse the _graph to get shortest path. The target gets scaned for all four incoming directions.
        :param start_position: 2d coordinate
        :param start_direction: orientation passed a agent moving direction
        :param target_position: 2d coordinate of target position
        :return: shortest path [edge[resource],..., edge[...]], all paths, total nbr of resource (len) of all found paths
        '''
        from_edge = self.get_mapped_vertex(start_position, start_direction)
        paths = []
        paths_len = []
        for target_direction in range(4):
            path = []
            to_edge = self.get_mapped_vertex(target_position, target_direction)
            path_len = 0
            if from_edge is not None and to_edge is not None:
                from_node_intern = from_edge[0]
                to_node_intern = to_edge[1]
                fn = self.get_graph().nodes.get(from_node_intern, None)
                tn = self.get_graph().nodes.get(to_node_intern, None)

                if fn is not None and tn is not None:
                    try:
                        s_path = nx.shortest_path(self.get_graph(), from_node_intern, to_node_intern, weight=weight)
                        start_p = from_node_intern
                        for p in s_path:
                            if start_p != p:
                                # xy, _ = FlatlandGraphBuilder.get_coordinate_direction_from_node_id(p)
                                resources = self.get_graph().get_edge_data(start_p, p).get('resources')
                                path.append(resources)
                                path_len += len(resources)
                            start_p = p
                    except nx.NetworkXNoPath:
                        pass
            paths.append(path)
            if path_len == 0 and not fast_position_equal(start_position, target_position):
                path_len = np.inf
            paths_len.append(path_len)
        arg_sorted_path_len = np.argsort(paths_len)
        return paths[arg_sorted_path_len[0]], paths, paths_len

    def prepare_observation_data_plot(self, start_position, start_direction, target_position, weight=None):
        '''
        Prepare the classified cells (railroad_switches, railroad_switch_neighbours) to render. The rendering used
        the observation rendering data structure.
        :param start_position: 2d coordinate
        :param start_direction: orientation passed a agent moving direction
        :param target_position: 2d coordinate of target position
        :return:
        '''
        path, paths, paths_len = self.get_shortest_path(start_position, start_direction, target_position, weight)

        env = self.railroad_switch_analyser.get_rail_env()
        arg_sorted_path_len = np.argsort(paths_len)

        for h in env.get_agent_handles():
            env.dev_obs_dict.update({h: []})
            if h < 4:
                env.dev_obs_dict.update({h: paths[arg_sorted_path_len[h]]})