# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
from dace.transformation.dataflow.strip_mining import StripMining
from typing import Any, Dict, Iterable, List, Set, Tuple, Union

import networkx
from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import propagation, utils, graph
from dace.codegen.targets import fpga
from dace.transformation import transformation, interstate
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet, data
import math


def modify_bank_assignment(array_name: str,
                           sdfg: SDFG,
                           new_memory: str,
                           new_bank: str,
                           split_array_info: List[int] = None):
    """
        Updates bank assignments for the array on the SDFG. Will update 
        the shape of the array as well depending on the previous assignment.
        :param split_array_info: A list with the same length as the old dimension 
        of the array. When transfering to HBM the size in each dimension is divided by
        the corresponding int, when moving to DDR it is multiplied. 
        """
    desc = sdfg.arrays[array_name]
    old_memory = None
    if 'memorytype' in desc.location and desc.location["memorytype"] is not None:
        old_memory = desc.location["memorytype"]
    if new_memory == "HBM":
        low, high = fpga.get_multibank_ranges_from_subset(new_bank, sdfg)
    else:
        low, high = int(new_bank), int(new_bank) + 1
    if split_array_info is None:
        d_size = len(desc.shape)
        if fpga.is_hbm_array_with_distributed_index(desc):
            d_size -= 1
        split_array_info = [1] * d_size

    if (old_memory is None or old_memory == "DDR") and new_memory == "HBM":
        desc = sdfg.arrays[array_name]
        new_shape = [x // y for x, y in zip(desc.shape, split_array_info)]
        if high - low > 1:
            desc.set_shape((high - low, *new_shape))
        else:
            desc.set_shape(new_shape)
    elif old_memory == "HBM" and (new_memory == "DDR" or new_memory is None):
        desc = sdfg.arrays[array_name]
        if fpga.is_hbm_array_with_distributed_index(desc):
            old_shape = list(desc.shape)[1:]
        else:
            old_shape = desc.shape
        new_shape = [x * y for x, y in zip(old_shape, split_array_info)]
        desc.set_shape(new_shape)
    elif old_memory == "HBM" and new_memory == "HBM":
        oldlow, oldhigh = fpga.get_multibank_ranges_from_subset(
            desc.location["bank"], sdfg)
        if oldlow == low and oldhigh == high:
            return
        # It would be problematic to change the number of banks, because of split_array_info
        raise NotImplementedError("Cannot directly transfer from HBM to HBM")
    desc.location["memorytype"] = new_memory
    desc.location['bank'] = new_bank
    desc.storage = dtypes.StorageType.FPGA_Global


def _multiply_sdfg_executions(sdfg: SDFG, outer_map_range: Tuple[str, int]):
    """
        Nests a whole SDFG and packs it into an unrolled map. 
        Depending on the values in update_array_access the first
        index of inputs/outputs is changed to the map param.
        """
    nesting = interstate.NestSDFG(sdfg.sdfg_id, -1, {}, -1)
    nesting.apply(sdfg)
    state = sdfg.states()[0]
    nsdfg_node = list(
        filter(lambda x: isinstance(x, nd.NestedSDFG), state.nodes()))[0]

    map_enter, map_exit = state.add_map(
        "hbm_unrolled_map", {outer_map_range[0]: f"0:{outer_map_range[1]}"},
        dtypes.ScheduleType.Unrolled)

    for input in state.in_edges(nsdfg_node):
        state.remove_edge(input)
        state.add_memlet_path(input.src,
                              map_enter,
                              nsdfg_node,
                              memlet=input.data,
                              src_conn=input.src_conn,
                              dst_conn=input.dst_conn)
    for output in state.out_edges(nsdfg_node):
        state.remove_edge(output)
        state.add_memlet_path(nsdfg_node,
                              map_exit,
                              output.dst,
                              memlet=output.data,
                              src_conn=output.src_conn,
                              dst_conn=output.dst_conn)


def _update_memlet_hbm(state: SDFGState, inner_edge: graph.MultiConnectorEdge,
                       inner_subset_index: symbolic.symbol,
                       this_node: nd.AccessNode):
    """
        Add the subset_index to the memlet path defined by convertible_node. If the end/start of
        the path is also an AccessNode, it will insert a tasklet before the access to 
        avoid validation failures due to dimensionality mismatch.
        :param inner_edge: The inner_edge of the path to modify
        :param inner_subset_index: The distributed subset for the innermost edge on
            the memlet path defined by convertible_node
        :param this_node: The AccessNode for HBM which is associated with this call to 
            the function. (i.e. one side of the path)
        """
    mem: memlet.Memlet = inner_edge.data
    # If the memlet already contains the distributed subset, ignore it
    # That's helpful because of inconsistencies when nesting and because
    # one can 'hint' the correct bank assignment when using this function
    if len(mem.subset) == len(state.parent.arrays[this_node.data].shape):
        return
    new_subset = subsets.Range([[inner_subset_index, inner_subset_index, 1]] +
                               [x for x in mem.subset])

    path = state.memlet_path(inner_edge)
    if path[-1].dst == this_node:
        is_write = True
        other_node = path[0].src
    elif path[0].src == this_node:
        is_write = False
        other_node = path[-1].dst

    if isinstance(other_node, nd.NestedSDFG): # Ignore those and update them via propagation
        new_subset = subsets.Range.from_array(state.parent.arrays[this_node.data])

    if isinstance(other_node, nd.AccessNode):
        fwtasklet = state.add_tasklet("fwtasklet", set(["_in"]), set(["_out"]),
                                      "_out = _in")
        state.remove_edge(inner_edge)
        target_other_subset = mem.other_subset
        mem.other_subset = None
        if is_write:
            inner_edge = state.add_edge(fwtasklet, '_out', inner_edge.dst,
                                        inner_edge.dst_conn, mem)
            state.add_edge(
                other_node, path[0].src_conn, fwtasklet, "_in",
                memlet.Memlet(other_node.data, subset=target_other_subset))
        else:
            inner_edge = state.add_edge(inner_edge.src, inner_edge.src_conn,
                                        fwtasklet, '_in', mem)
            state.add_edge(
                fwtasklet, "_out", other_node, path[-1].dst_conn,
                memlet.Memlet(other_node.data, subset=target_other_subset))

    inner_edge.data.subset = new_subset


def _update_new_hbm_accesses(sdfg: SDFG,
                             update_access: set(),
                             inner_subset_index: symbolic.symbol,
                             recursive=True):
    """
    Update all acccesses to multibank-arrays.
    :param update_access: The names of new multibank-arrays
    :param inner_subset_index: The name of the map variable
    :param recursive: Check also in nested SDFGs
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nd.AccessNode) and node.data in update_access:
                for inner_edge in utils.all_innermost_edges(state, node):
                    _update_memlet_hbm(state, inner_edge, inner_subset_index,
                                       node)


def _recursive_hbm_transform(sdfg, inner_subset_index, update_array_banks, recursive):
    update_access = set()  # Store which arrays need updates for later

    # update array bank positions
    for array_name, infos in update_array_banks.items():
        memory_type, bank, divide_shape = infos
        if array_name in sdfg.arrays:
            modify_bank_assignment(array_name, sdfg, memory_type, bank,
                                divide_shape)
        if memory_type == "HBM":
            low, high = fpga.get_multibank_ranges_from_subset(bank, sdfg)
            if high - low > 1:
                update_access.add(array_name)
    
    _update_new_hbm_accesses(sdfg, update_access, inner_subset_index)

    if not recursive:
        return

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nd.NestedSDFG):
                inner_banks = {}
                node.symbol_mapping[str(
                    inner_subset_index)] = inner_subset_index

                def add_pass_update(inner_name, outer_name):
                    if outer_name in update_array_banks:
                        inner_banks[inner_name] = update_array_banks[outer_name]

                for edge in state.in_edges(node):
                    add_pass_update(edge.dst_conn, edge.data.data)
                for edge in state.out_edges(node):
                    add_pass_update(edge.src_conn, edge.data.data)
                _recursive_hbm_transform(node.sdfg, inner_subset_index, inner_banks, True)


def transform_sdfg_for_hbm(sdfg: SDFG,
                           outer_map_range: Tuple[str, int],
                           update_array_banks: Dict[str, Tuple[str, str,
                                                               List[int]]],
                           update_map_range: Dict[Tuple[nd.Map, int], int],
                           recursive=False):
    """
    This function is a tool which allows to quickly rewrite SDFGs to use many HBM-banks. 
    Essentially all it does is nest the whole SDFG and pack it into a top-level unrolled map. 
    Additionally it contains options to change the bank assignment of arrays and to modify accesses 
    such that they contain the top-level unrolled map variable as a distributed subset (i.e. as 
    an additional first index). 
    This makes it also usefull to quickly switch bank assignments of existing arrays and have
    stuff like dimensionality change be handled automatically.
    Note that this expects to be used on an SDFG which will run on the FPGA.
    :param outer_map_range: A tuple of (distributed subset variable name, w), where the top level map
        will count from 0 to w
    :param update_array_banks: dict from array name to a tuple of (new memorytype, new bank(s), how
        to divide/multiply the old shape of the array). The tuple actually represents the parameters for
        modify_bank_assignments, so look there for more explanation.
    :param update_map_range: A dict from tuples of (map, index of the symbol on the map) to the 
        value with which the range of that map should be divided. Using this makes sense if you
        would like to decrease the times a map executes such that the total elements processed 
        stay the same.
    :param recursive: Also look inside Nested SDFGs
    """

    for map_info, division in update_map_range.items():
        target, param_index = map_info
        current = target.range[param_index]
        new_value = (
            current[0],
            symbolic.pystr_to_symbolic(f"{current[1] + 1}//{division} - 1"),
            current[2])
        target.range[param_index] = new_value

    # We need to update on the inner part as well - if recursive is false one needs to do so explicit
    if not recursive:
        _recursive_hbm_transform(sdfg, outer_map_range[0], update_array_banks, False)

    # nest the sdfg and execute in parallel
    _multiply_sdfg_executions(sdfg, outer_map_range)

    # Update array assignments and accesses
    _recursive_hbm_transform(sdfg, outer_map_range[0], update_array_banks, recursive)

    # set default on all outer arrays, such that FPGATransformSDFG can be used
    for desc in sdfg.arrays.items():
        desc[1].storage = dtypes.StorageType.Default

    # memlets will be inconsistent after that, so propagate
    propagation.propagate_memlets_sdfg(sdfg)

@registry.autoregister_params(singlestate=True)
@properties.make_properties
class HbmTransform(transformation.Transformation):

    _map_entry = nd.MapEntry(nd.Map("", [], []))

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int], expr_index: int,
                       sdfg: SDFG, strict: bool) -> bool:
        if strict:
            return False

        # This must run on on-device code
        if not isinstance(graph, SDFGState) or not fpga.can_run_state_on_fpga(graph):
            return False

        map_entry = graph.nodes()[candidate[HbmTransform._map_entry]]
        map_exit = graph.exit_node(map_entry)

        # Can't handle nesting
        scope = graph.scope_subgraph(map_entry)
        for node in scope.nodes():
            if isinstance(node, nd.NestedSDFG) or isinstance(node, nd.LibraryNode):
                return False

        if len(map_entry.map.params) != 1:
            return False

        # Check if all arrays are assigned, and we can somehow split
        if HbmTransform._scan_paths(sdfg, graph, map_entry, map_exit) is None:
            return False

        return True

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(HbmTransform._map_entry)
        ]

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        state: SDFGState = sdfg.nodes()[self.state_id]
        unroll_entry: nd.MapEntry = state.nodes()[self.subgraph[self._map_entry]]
        unroll_exit = state.exit_node(unroll_entry)

        split_arrays, no_split_arrays, unroll_factor, split_dimensions, array_dimensions = HbmTransform._scan_paths(
            sdfg, state, unroll_entry, unroll_exit,
        )

        new_map: nd.Map = StripMining.apply_to(sdfg, 
        {"tile_size": unroll_factor, "divides_evenly": True, "skew": True, "tiling_type": dtypes.TilingType.CeilRange,
            "new_dim_prefix": "bank"},
        _map_entry=unroll_entry)
        for n in state.nodes():
            if isinstance(n, nd.MapEntry) and n.map == new_map:
                #nodes are turned around by strip mine
                inner_entry = unroll_entry
                unroll_entry = n
                unroll_exit = state.exit_node(n)
                break

        # Switch the maps, update schedules, set outer parameter
        tmp_inner_param = inner_entry.map.params[0]
        tmp_to_inner_range = unroll_entry.map.range[0]
        tmp_to_outer_range = inner_entry.map.range[0]
        tmp_old_outer_param = unroll_entry.map.params[0]
        scope_view = state.scope_subgraph(unroll_entry)

        new_dim = "k"
        unroll_entry.map.params[0] = new_dim
        unroll_entry.map.range[0] = tmp_to_outer_range
        inner_entry.map.range[0] = tmp_to_inner_range
        inner_entry.map.schedule = dtypes.ScheduleType.Default
        unroll_entry.map.schedule = dtypes.ScheduleType.Unrolled

        # We remove the multiplication (since it's not needed any more) on the old parameter,
        # but keep it so we can easier replace with the new dimension later
        scope_view.replace(tmp_old_outer_param, f"({tmp_old_outer_param}/{unroll_factor})")

        # Actually place the arrays and update paths
        for edge in state.in_edges(unroll_entry) + state.out_edges(unroll_exit):
            name = edge.data.data
            if name in split_arrays or name in no_split_arrays:
                desc = sdfg.arrays[name]
                memory_type = desc.location["memorytype"]
                desc.location.pop("memorytype")
                bank = desc.location["bank"]
                desc.location.pop("bank")
                if name in split_arrays:
                    division_info = [1] * array_dimensions[name]
                    division_info[split_dimensions[name]] = unroll_factor
                else:
                    division_info = None
                modify_bank_assignment(name, sdfg, memory_type, bank, division_info)

            if name in split_arrays:
                path = state.memlet_path(edge)
                if isinstance(path[0].src, nd.AccessNode) and path[0].src.data == name:
                    this_node = path[0].src
                    inner_edge = path[-1]
                else:
                    this_node = path[-1].dst
                    inner_edge = path[0]
                
                current_sbs = inner_edge.data.subset[split_dimensions[name]]
                new_value = symbolic.pystr_to_symbolic(f"{current_sbs[0]}-{tmp_old_outer_param}")
                inner_edge.data.subset[split_dimensions[name]] = (new_value, new_value, current_sbs[2])
                _update_memlet_hbm(state, inner_edge, new_dim, this_node)

        # Replace the dummy symbol everywhere where it still remains
        scope_view.replace(tmp_old_outer_param, f"({tmp_to_inner_range[1]}+1)*{new_dim}")

        # Propagate the modified inner memlets
        propagation.propagate_memlets_state(sdfg, state)


    @staticmethod
    def _scan_paths(sdfg: SDFG, state: SDFGState, map_entry: nd.MapEntry, map_exit: nd.MapExit):
        """
        Find all arrays and record them if they are placed in global memory.
        Find present bank assignemnts and constraints for allowed unroll factor for all arrays based on shape.
        :return: A tuple of (arrays in global memory, arrays which may not be split based on their assignment,
            an unroll factor which is set if an array is placed on multiple HBM-banks, 
            a dict of arrays which are pre assigned to some location,
            a list from arrays to the values a potential split has to divide, 
            the dimensions of the array in global memory)
        """

        unroll_factor = None
        no_split_arrays = {}
        split_arrays = {}
        array_dimensions = {}
        split_dimensions = {}

        attached_array = {}
        for edge in state.in_edges(map_entry) + state.out_edges(map_exit):
            if edge.data.is_empty():
                continue
            if edge.data.data in attached_array: # Only one edge per array
                return None
            attached_array[edge.data.data] = edge

        for name in attached_array:
            desc = sdfg.arrays[name]

            if not isinstance(desc, data.Array) or isinstance(desc, data.View):
                continue
            if desc.storage != dtypes.StorageType.FPGA_Global and desc.storage != dtypes.StorageType.Default:  # If not in global memory ignore
                continue

            assigned = fpga.parse_location_bank(desc)
            if assigned is None: # All arrays must be assigned
                return None 
            else:
                if assigned[0] == "HBM":
                    low, high = fpga.get_multibank_ranges_from_subset(
                        assigned[1], sdfg)
                    if high - low ==  1:
                        no_split_arrays[name] = assigned
                        continue
                    if unroll_factor is None:
                        unroll_factor = high - low
                    else:
                        if unroll_factor != high - low: # All split arrays must have the same number of banks
                            return None
                    split_arrays[name] = assigned
                else:
                    no_split_arrays[name] = assigned
            array_dimensions[name] = len(desc.shape)

        # Check if the arrays which should be split can do so
        for name in split_arrays:
            edge = attached_array[name]
            count_innermost = 0
            for edge in utils.all_innermost_edges(state, edge):
                count_innermost += 1
                if count_innermost > 1:
                    return None # Can't handle trees
                innermost = edge
            
            found = None
            for i, val in enumerate(innermost.data.subset):
                low, high, stride = val
                if stride != 1 or low != high:
                    continue
                if map_entry.map.params[0] in set([str(x) for x in low.free_symbols]):
                    if found is None:
                        found = i
                    else:
                        return None # Only 1 dimension may be dependent. 
            if found is None:
                return None
            
            # We assume that it the found dimension behaves linear in the map symbol
            split_dimensions[name] = found

        return (split_arrays, no_split_arrays, unroll_factor, split_dimensions, array_dimensions)
