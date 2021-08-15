# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.transformation.dataflow.strip_mining import StripMining
from typing import Any, Dict, List, Union

from dace import dtypes, properties, registry, subsets, symbolic
from dace.sdfg import propagation, utils, graph
from dace.codegen.targets import fpga
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace import SDFG, SDFGState, memlet, data


def modify_bank_assignment(array_name: str,
                           sdfg: SDFG,
                           new_memory: str,
                           new_bank: str,
                           split_array_info: List[int] = None,
                           set_storage_type: bool = True):
    """
        Updates bank assignments for the array on the SDFG. Will update 
        the shape of the array as well depending on the previous assignment.
        :param split_array_info: A list with the same length as the old dimension 
        of the array. When transfering to HBM the size in each dimension is divided by
        the corresponding int, when moving to DDR it is multiplied. 
        :param set_storage_type: Place the array on FPGA_Global.
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
    if set_storage_type:
        desc.storage = dtypes.StorageType.FPGA_Global


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

    if isinstance(
            other_node,
            nd.NestedSDFG):  # Ignore those and update them via propagation
        new_subset = subsets.Range.from_array(
            state.parent.arrays[this_node.data])

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


@registry.autoregister_params(singlestate=True)
@properties.make_properties
class HbmTransform(transformation.Transformation):
    """
    A transformation that applies on a map when all attached global memories are
    assigned to banks. Note that the assignment is rather a hinting, the actual
    dimensionality changes (when) required will be done by the transformation.
    All arrays that should span across multiple banks (i.e. should be split) must have
    exactly one dimension where an access happens in dependence of the map variable i.
    All attached arrays must either be assigned to one single bank or span across
    the same number of banks. 

    Moves all attached arrays to their banks, and changes their size according to the
    bank assignment. Adds an outer unrolled loop around the map on which it is applied
    and changes all accesses such that they go to the same place as before (for single
    bank assignments), respectively to the right bank and the modified location for
    multibank arrays. 

    For any access in some dimension of an array dependent on the map variable i the transformation
    assumes that the position behaves linear in i. Also in such dimension the array size has to be
    divisable by the number of banks across which all multibank arrays span.

    At the moment the transformation cannot apply if the same array
    in global memory is attached to the map with multiple edges. This also implies
    that write-back to an array from which data is read is disallowed. It is done this way
    because it reduces the complexity of the transformation somewhat and because in
    the context of HBM this is likely a bad idea anyway. (More complex routing + reduced IO).
    """

    _map_entry = nd.MapEntry(nd.Map("", [], []))

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [utils.node_path_graph(HbmTransform._map_entry)]

    new_dim = properties.Property(
        dtype=str,
        default="k",
        desc="Defines the map param of the outer unrolled map")

    move_to_FPGA_global = properties.Property(
        dtype=bool,
        default=True,
        desc="All assigned arrays have their storage changed to  FPGA_Global")

    @staticmethod
    def can_be_applied(graph: Union[SDFG, SDFGState],
                       candidate: Dict['PatternNode', int], expr_index: int,
                       sdfg: SDFG, strict: bool) -> bool:
        if strict:
            return False

        # This must run on on-device code
        if not isinstance(graph,
                          SDFGState) or not fpga.can_run_state_on_fpga(graph):
            return False

        map_entry = graph.nodes()[candidate[HbmTransform._map_entry]]
        map_exit = graph.exit_node(map_entry)

        # Can't handle nesting
        scope = graph.scope_subgraph(map_entry)
        for node in scope.nodes():
            if isinstance(node, nd.NestedSDFG) or isinstance(
                    node, nd.LibraryNode):
                return False

        if len(map_entry.map.params) != 1:
            return False

        # Check if all arrays are assigned, and we can somehow split
        result = HbmTransform._scan_paths(sdfg, graph, map_entry, map_exit)
        if result is None:
            return False

        return True

    def apply(self, sdfg: SDFG) -> Union[Any, None]:
        state: SDFGState = sdfg.nodes()[self.state_id]
        unroll_entry: nd.MapEntry = state.nodes()[self.subgraph[
            self._map_entry]]
        unroll_exit = state.exit_node(unroll_entry)

        split_arrays, no_split_arrays, unroll_factor, split_dimensions = HbmTransform._scan_paths(
            sdfg,
            state,
            unroll_entry,
            unroll_exit,
        )

        tile_prefix = sdfg._find_new_name("bank")
        new_map: nd.Map = StripMining.apply_to(
            sdfg, {
                "tile_size": unroll_factor,
                "divides_evenly": True,
                "skew": True,
                "tiling_type": dtypes.TilingType.CeilRange,
                "new_dim_prefix": tile_prefix
            },
            _map_entry=unroll_entry)
        for n in state.nodes():
            if isinstance(n, nd.MapEntry) and n.map == new_map:
                #nodes are turned around by strip mine
                inner_entry = unroll_entry
                unroll_entry = n
                unroll_exit = state.exit_node(n)
                break

        # Switch the maps, update schedules, set outer parameter
        tmp_to_inner_range = unroll_entry.map.range[0]
        tmp_to_outer_range = inner_entry.map.range[0]
        tmp_old_outer_param = unroll_entry.map.params[0]
        scope_view = state.scope_subgraph(unroll_entry)

        unroll_entry.map.params[0] = self.new_dim
        unroll_entry.map.range[0] = tmp_to_outer_range
        inner_entry.map.range[0] = tmp_to_inner_range
        inner_entry.map.schedule = dtypes.ScheduleType.Default
        unroll_entry.map.schedule = dtypes.ScheduleType.Unrolled

        # We remove the multiplication (since it's not needed any more) on the old parameter,
        # but keep it so we can easier replace with the new dimension later
        scope_view.replace(tmp_old_outer_param,
                           f"({tmp_old_outer_param}/{unroll_factor})")

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
                    division_info = [1] * len(sdfg.arrays[name].shape)
                    division_info[split_dimensions[name]] = unroll_factor
                else:
                    division_info = None
                modify_bank_assignment(name, sdfg, memory_type, bank,
                                       division_info, self.move_to_FPGA_global)

            if name in split_arrays:
                path = state.memlet_path(edge)
                if isinstance(path[0].src,
                              nd.AccessNode) and path[0].src.data == name:
                    this_node = path[0].src
                    inner_edge = path[-1]
                else:
                    this_node = path[-1].dst
                    inner_edge = path[0]

                inner_edge.data.replace({tmp_old_outer_param: "0"})
                _update_memlet_hbm(state, inner_edge, self.new_dim, this_node)

        # Replace the dummy symbol everywhere where it still remains
        scope_view.replace(tmp_old_outer_param,
                           f"({tmp_to_inner_range[1]}+1)*{self.new_dim}")

        # Propagate the modified inner memlets
        propagation.propagate_memlets_state(sdfg, state)

    @staticmethod
    def _scan_paths(sdfg: SDFG, state: SDFGState, map_entry: nd.MapEntry,
                    map_exit: nd.MapExit):
        """
        Find all arrays attached to the map, check their bank assignment/accesses
        and find a suitable unroll factor if possible
        :return: A tuple of (split_arrays, no_split_arrays, unroll_factor, split_dimensions,
                array_dimensions), where split_arrays the array names that are split,
                no_split_arrays array names that are not split,
                unroll_factor the value for the number of splits that are created from
                    split_arrrays,
                split_dimensions a mapping from array name to the dimension along which
                the array should be split (always only 1).
        """

        unroll_factor = None
        no_split_arrays = {}
        split_arrays = {}
        split_dimensions = {}
        has_pending_changes = False  # Will there something be done?

        attached_array = {}
        for edge in state.in_edges(map_entry) + state.out_edges(map_exit):
            if edge.data.is_empty():
                continue
            if edge.data.data in attached_array:  # Only one edge per array
                return None
            attached_array[edge.data.data] = edge

        for name in attached_array:
            desc = sdfg.arrays[name]

            if not isinstance(desc, data.Array) or isinstance(desc, data.View):
                continue
            if desc.storage != dtypes.StorageType.FPGA_Global and desc.storage != dtypes.StorageType.Default:  # If not in global memory ignore
                continue

            assigned = fpga.parse_location_bank(desc)
            if assigned is None:  # All arrays must be assigned
                return None
            else:
                if assigned[0] == "HBM":
                    low, high = fpga.get_multibank_ranges_from_subset(
                        assigned[1], sdfg)
                    if high - low == 1:
                        no_split_arrays[name] = assigned
                        continue
                    if unroll_factor is None:
                        unroll_factor = high - low
                    else:
                        if unroll_factor != high - low:  # All split arrays must have the same number of banks
                            return None
                    split_arrays[name] = assigned

                    # Otherwise we assume the array was already placed
                    if desc.shape[0] != high - low:
                        has_pending_changes = True
                    else:
                        return None  # If an array was already placed on HBM we cannot apply
                else:
                    no_split_arrays[name] = assigned

        # Check if the arrays which should be split can do so
        for name in split_arrays:
            edge = attached_array[name]
            count_innermost = 0
            for edge in utils.all_innermost_edges(state, edge):
                count_innermost += 1
                if count_innermost > 1:
                    return None  # Can't handle trees
                innermost = edge

            found = None
            for i, val in enumerate(innermost.data.subset):
                low, high, stride = val
                if stride != 1 or low != high:
                    continue
                if map_entry.map.params[0] in set(
                    [str(x) for x in low.free_symbols]):
                    if found is None:
                        found = i
                    else:
                        return None  # Only 1 dimension may be dependent.
            if found is None:
                return None

            # We assume that it the found dimension behaves linear in the map symbol
            split_dimensions[name] = found

        if not has_pending_changes:  # In this case we would do nothing
            return None

        return (split_arrays, no_split_arrays, unroll_factor, split_dimensions)
