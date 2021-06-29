from typing import Any, Dict, List, Tuple, Union

import networkx
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import utils, graph
from dace.transformation import transformation, interstate
from dace.sdfg import scope, nodes as nd
from dace import SDFG, SDFGState, memlet

@registry.autoregister
@properties.make_properties
class HbmTransform(transformation.Transformation):

    #dtype=List[Tuple[SDFGState, Union[nd.AccessNode, graph.MultiConnectorEdge], symbolic.symbol, bool)]]
    updated_access_list = properties.Property(
        dtype=List,
        default=[],
        desc=("List of edges defining memlet paths that are now accessing the specified subset "
            "index, together with the state they are part of. If there is only one path associated "
            "with an AccessNode, this may also be used to specify which path to modify. "
            "If the bool is set to true, a tasklet will be used to write/read the final value "
            "so it bypasses SDFG dimensionality validation.")
    )

    #dtype=List[Tuple[str, str]]
    updated_array_list = properties.Property(
        dtype=List,
        default=[],
        desc="List of (arrayname, new value for location['bank']). Will also update the shape of the array."
    )

    outer_map_range = properties.Property(
        dtype=Dict,
        default={"k" : "0"},
        desc="Stores the range for the outer HBM map. defaults to k = 0."
    )

    def _multiply_sdfg_executions(self, sdfg : SDFG, 
        unrollparams : Union[Dict[str, str], List[Tuple[str, str]]]):
        """
        Nests a whole SDFG and packs it into an unrolled map
        """
        nesting = interstate.NestSDFG(sdfg.sdfg_id, -1, {}, self.expr_index)
        nesting.apply(sdfg)
        state = sdfg.states()[0]
        nsdfg_node = list(filter(lambda x : isinstance(x, nd.NestedSDFG), state.nodes()))[0]
        
        map_enter, map_exit = state.add_map("hbm_unrolled_map", unrollparams, 
            dtypes.ScheduleType.Unrolled)

        for input in list(state.in_edges(nsdfg_node)):
            state.remove_edge(input)
            state.add_memlet_path(input.src, map_enter, nsdfg_node,
                memlet=input.data, src_conn=input.src_conn, 
                dst_conn=input.dst_conn)
        for output in list(state.out_edges(nsdfg_node)):
            state.remove_edge(output)
            state.add_memlet_path(nsdfg_node, map_exit, output.dst,
                memlet=output.data, src_conn=output.src_conn, 
                dst_conn=output.dst_conn)

    def _update_memlet_hbm(self, state : SDFGState, 
        inner_edge : graph.MultiConnectorEdge, inner_subset_index : symbolic.symbol,
        use_final_tasklet : bool):
        """
        Add the subset_index to the memlet path defined by inner_edge
        """
        mem : memlet.Memlet = inner_edge.data
        mem.subset = subsets.Range([[inner_subset_index, inner_subset_index, 1]] +
            [x for x in mem.subset])
        
        path = state.memlet_path(inner_edge)
        edge_index = path.index(inner_edge)
        if edge_index == 0:
            is_write=True
        elif edge_index == len(path)-1:
            is_write=False
        else:
            raise ValueError("The provided edge is not the innermost")
        src_conn = path[0].src_conn
        dst_conn = path[-1].dst_conn
        path_nodes = []
        for edge in path:
            path_nodes.append(edge.src)
            state.remove_edge_and_connectors(edge)
        path_nodes.append(path[-1].dst)

        if use_final_tasklet:
            fwtasklet = state.add_tasklet(
                "fwtasklet", set(["_in"]), set(["_out"]),
                "_out = _in"
            )
            if is_write:
                overwritten_node = path_nodes[0]
                path_nodes[0] = fwtasklet                
                src_conn = "_out"
            else:
                overwritten_node = path_nodes[-1]
                path_nodes[-1] = fwtasklet
                dst_conn = "_in"

        state.add_memlet_path(*path_nodes, memlet=mem,
            src_conn=src_conn, dst_conn=dst_conn)

        if use_final_tasklet:
            if is_write:
                state.add_memlet_path(overwritten_node, fwtasklet, 
                    memlet=memlet.Memlet(mem.data), 
                    src_conn=path[0].src_conn, dst_conn="_in",)
            else:
                state.add_memlet_path(fwtasklet, overwritten_node,
                    memlet=memlet.Memlet(mem.data), 
                    src_conn="_out", dst_conn=path[-1].dst_conn,)

        
    @staticmethod
    def can_be_applied(self, graph: Union[SDFG, SDFGState], candidate: Dict['PatternNode', int],
        expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [networkx.DiGraph()]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        for path_desc_state, path_description, subset_index, final_tasklet in self.updated_access_list:
            if isinstance(path_description, nd.AccessNode):
                some_edge = list(path_desc_state.all_edges(path_description))
                if len(some_edge) != 1:
                    raise ValueError("You may not specify an AccessNode in the update_access_list "
                        " if it does not have exactly one attached edge")
                some_edge = some_edge[0]
                if some_edge.dst == path_description:
                    path_description = path_desc_state.memlet_path(some_edge)[0]
                else:
                    path_description = path_desc_state.memlet_path(some_edge)[-1]
            
            self._update_memlet_hbm(path_desc_state, path_description, subset_index, final_tasklet)

        for array_name, bankprop in self.updated_array_list:
            desc = sdfg.arrays[array_name]
            desc.location['bank'] = bankprop
            tmp = utils.parse_location_bank(desc)
            if tmp is None:
                raise ValueError(f"Could not parse {bankprop} in property updated_array_list")
            low, high = utils.get_multibank_ranges_from_subset(tmp[1], sdfg)
            desc.shape = (high - low, *desc.shape)
   
        self._multiply_sdfg_executions(sdfg, self.outer_map_range)
        sdfg.view()


