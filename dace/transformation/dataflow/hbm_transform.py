from typing import Any, Dict, List, Tuple, Union

import networkx
from dace import dtypes, registry
from dace.sdfg import utils
from dace.transformation import transformation, interstate
from dace.sdfg import scope, nodes as nd
from dace import SDFG, SDFGState

class HbmTransform(transformation.Transformation):
    def _multiply_sdfg_executions(self, sdfg : SDFG, 
        unrollparams : Union[Dict[str, str], List[Tuple[str, str]]]):
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
        
        #sdfg.view()

        sdfg.validate()
        pass

        

    @staticmethod
    def can_be_applied(self, graph: Union[SDFG, SDFGState], candidate: Dict['PatternNode', int],
        expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        return True

    @staticmethod
    def expressions():
        # Matches anything
        return [networkx.DiGraph()]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        self._multiply_sdfg_executions(sdfg, [("k", "0:5")])
