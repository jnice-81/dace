# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Exception classes and methods for validation of SDFGs. """
import copy
from dace.sdfg.nodes import Tasklet
from threading import local

from numpy import isin, true_divide
from dace.dtypes import StorageType
import os
from typing import Dict, Tuple, Union
import warnings

from dace import dtypes, memlet, data
from dace import symbolic

#TODO: HBM
        #Checks:
        #Hbm multibank can be parsed
        #magic index exists for multibanks resp. not exists for single banks (dimension check)
        #Update dst and src subset check to support divison: TODO -> volume not checked at the moment
        #Check that magic index is symbolic fixed bound or integer and in bounds
        #no multibank memlet is attached to a tasklet: part of the subset check
        #TODO: Check that strides are ok, no left out elements (only simple 2d/3d allowed)

###########################################
# Validation

def validate(graph: 'dace.sdfg.graph.SubgraphView'):
    from dace.sdfg import SDFG, SDFGState, SubgraphView
    gtype = graph.parent if isinstance(graph, SubgraphView) else graph
    if isinstance(gtype, SDFG):
        validate_sdfg(graph)
    elif isinstance(gtype, SDFGState):
        validate_state(graph)


def validate_sdfg(sdfg: 'dace.sdfg.SDFG'):
    """ Verifies the correctness of an SDFG by applying multiple tests.
        :param sdfg: The SDFG to verify.

        Raises an InvalidSDFGError with the erroneous node/edge
        on failure.
    """
    try:
        # SDFG-level checks
        if not dtypes.validate_name(sdfg.name):
            raise InvalidSDFGError("Invalid name", sdfg, None)

        if len(sdfg.source_nodes()) > 1 and sdfg.start_state is None:
            raise InvalidSDFGError("Starting state undefined", sdfg, None)

        if len(set([s.label for s in sdfg.nodes()])) != len(sdfg.nodes()):
            raise InvalidSDFGError("Found multiple states with the same name",
                                   sdfg, None)

        # Validate data descriptors
        for name, desc in sdfg._arrays.items():
            # Validate array names
            if name is not None and not dtypes.validate_name(name):
                raise InvalidSDFGError("Invalid array name %s" % name, sdfg,
                                       None)
            # Allocation lifetime checks
            if (desc.lifetime is dtypes.AllocationLifetime.Persistent
                    and desc.storage is dtypes.StorageType.Register):
                raise InvalidSDFGError(
                    "Array %s cannot be both persistent and use Register as "
                    "storage type. Please use a different storage location." %
                    name, sdfg, None)

        # Check every state separately
        start_state = sdfg.start_state
        symbols = copy.deepcopy(sdfg.symbols)
        symbols.update(sdfg.arrays)
        symbols.update(sdfg.constants)
        for desc in sdfg.arrays.values():
            for sym in desc.free_symbols:
                symbols[str(sym)] = sym.dtype
        visited = set()
        visited_edges = set()
        # Run through states via DFS, ensuring that only the defined symbols
        # are available for validation
        for edge in sdfg.dfs_edges(start_state):
            # Source -> inter-state definition -> Destination
            ##########################################
            visited_edges.add(edge)
            # Source
            if edge.src not in visited:
                visited.add(edge.src)
                validate_state(edge.src, sdfg.node_id(edge.src), sdfg, symbols)

            ##########################################
            # Edge
            # Check inter-state edge for undefined symbols
            undef_syms = set(edge.data.free_symbols) - set(symbols.keys())
            if len(undef_syms) > 0:
                eid = sdfg.edge_id(edge)
                raise InvalidSDFGInterstateEdgeError(
                    "Undefined symbols in edge: %s" % undef_syms, sdfg, eid)

            # Validate inter-state edge names
            issyms = edge.data.new_symbols(symbols)
            if any(not dtypes.validate_name(s) for s in issyms):
                invalid = next(s for s in issyms if not dtypes.validate_name(s))
                eid = sdfg.edge_id(edge)
                raise InvalidSDFGInterstateEdgeError(
                    "Invalid interstate symbol name %s" % invalid, sdfg, eid)

            # Add edge symbols into defined symbols
            symbols.update(issyms)

            ##########################################
            # Destination
            if edge.dst not in visited:
                visited.add(edge.dst)
                validate_state(edge.dst, sdfg.node_id(edge.dst), sdfg, symbols)
        # End of state DFS

        # If there is only one state, the DFS will miss it
        if start_state not in visited:
            validate_state(start_state, sdfg.node_id(start_state), sdfg,
                           symbols)

        # Validate all inter-state edges (including self-loops not found by DFS)
        for eid, edge in enumerate(sdfg.edges()):
            if edge in visited_edges:
                continue
            issyms = edge.data.assignments.keys()
            if any(not dtypes.validate_name(s) for s in issyms):
                invalid = next(s for s in issyms if not dtypes.validate_name(s))
                raise InvalidSDFGInterstateEdgeError(
                    "Invalid interstate symbol name %s" % invalid, sdfg, eid)

    except InvalidSDFGError as ex:
        # If the SDFG is invalid, save it
        sdfg.save(os.path.join('_dacegraphs', 'invalid.sdfg'), exception=ex)
        raise


def validate_state(state: 'dace.sdfg.SDFGState',
                   state_id: int = None,
                   sdfg: 'dace.sdfg.SDFG' = None,
                   symbols: Dict[str, dtypes.typeclass] = None):
    """ Verifies the correctness of an SDFG state by applying multiple
        tests. Raises an InvalidSDFGError with the erroneous node on
        failure.
    """
    # Avoid import loops
    from dace.sdfg import SDFG
    from dace.config import Config
    from dace.sdfg import nodes as nd
    from dace.sdfg.scope import scope_contains_scope
    from dace import data as dt
    from dace import subsets as sbs

    sdfg = sdfg or state.parent
    state_id = state_id or sdfg.node_id(state)
    symbols = symbols or {}

    if not dtypes.validate_name(state._label):
        raise InvalidSDFGError("Invalid state name", sdfg, state_id)

    if state._parent != sdfg:
        raise InvalidSDFGError("State does not point to the correct "
                               "parent", sdfg, state_id)

    # Unreachable
    ########################################
    if (sdfg.number_of_nodes() > 1 and sdfg.in_degree(state) == 0
            and sdfg.out_degree(state) == 0):
        raise InvalidSDFGError("Unreachable state", sdfg, state_id)

    #Correctly defined HBM-Arrays
    ########################################
    from dace.sdfg import hbm_multibank_expansion #Avoid circular import

    local_hbmmultibank_arrays = {}
    for name, array in sdfg.arrays.items():
        if("hbmbank" in array.location):
            try:
                currenthbmarray = hbm_multibank_expansion.parseHBMArray(name, array)
                if(currenthbmarray['splitcount'] > 1):
                    local_hbmmultibank_arrays[name] = currenthbmarray
            except Exception as e:
                raise InvalidSDFGError(f"Invalid HBM-info on array {name}: {str(e)}", sdfg, state_id)

    for nid, node in enumerate(state.nodes()):
        # Node validation
        try:
            node.validate(sdfg, state)
        except InvalidSDFGError:
            raise
        except Exception as ex:
            raise InvalidSDFGNodeError("Node validation failed: " + str(ex),
                                       sdfg, state_id, nid) from ex

        # Isolated nodes
        ########################################
        if state.in_degree(node) + state.out_degree(node) == 0:
            # One corner case: OK if this is a code node
            if isinstance(node, nd.CodeNode):
                pass
            else:
                raise InvalidSDFGNodeError("Isolated node", sdfg, state_id, nid)

        # Scope tests
        ########################################
        if isinstance(node, nd.EntryNode):
            try:
                state.exit_node(node)
            except StopIteration:
                raise InvalidSDFGNodeError(
                    "Entry node does not have matching "
                    "exit node",
                    sdfg,
                    state_id,
                    nid,
                )

        if isinstance(node, (nd.EntryNode, nd.ExitNode)):
            for iconn in node.in_connectors:
                if (iconn is not None and iconn.startswith("IN_")
                        and ("OUT_" + iconn[3:]) not in node.out_connectors):
                    raise InvalidSDFGNodeError(
                        "No match for input connector %s in output "
                        "connectors" % iconn,
                        sdfg,
                        state_id,
                        nid,
                    )
            for oconn in node.out_connectors:
                if (oconn is not None and oconn.startswith("OUT_")
                        and ("IN_" + oconn[4:]) not in node.in_connectors):
                    raise InvalidSDFGNodeError(
                        "No match for output connector %s in input "
                        "connectors" % oconn,
                        sdfg,
                        state_id,
                        nid,
                    )

        # Node-specific tests
        ########################################
        if isinstance(node, nd.AccessNode):
            if node.data not in sdfg.arrays:
                raise InvalidSDFGNodeError(
                    "Access node must point to a valid array name in the SDFG",
                    sdfg,
                    state_id,
                    nid,
                )
            arr = sdfg.arrays[node.data]

            # Verify View references
            if isinstance(arr, dt.View):
                from dace.sdfg import utils as sdutil  # Avoid import loops
                if sdutil.get_view_edge(state, node) is None:
                    raise InvalidSDFGNodeError(
                        "Ambiguous or invalid edge to/from a View access node",
                        sdfg, state_id, nid)

            # Find uninitialized transients
            if (arr.transient and state.in_degree(node) == 0
                    and state.out_degree(node) > 0
                    # Streams do not need to be initialized
                    and not isinstance(arr, dt.Stream)):
                # Find other instances of node in predecessor states
                states = sdfg.predecessor_states(state)
                input_found = False
                for s in states:
                    for onode in s.nodes():
                        if (isinstance(onode, nd.AccessNode)
                                and onode.data == node.data):
                            if s.in_degree(onode) > 0:
                                input_found = True
                                break
                    if input_found:
                        break
                if not input_found and node.setzero == False:
                    warnings.warn(
                        'WARNING: Use of uninitialized transient "%s" in state %s'
                        % (node.data, state.label))

            # Find writes to input-only arrays
            only_empty_inputs = all(e.data.is_empty()
                                    for e in state.in_edges(node))
            if (not arr.transient) and (not only_empty_inputs):
                nsdfg_node = sdfg.parent_nsdfg_node
                if nsdfg_node is not None:
                    if node.data not in nsdfg_node.out_connectors:
                        raise InvalidSDFGNodeError(
                            'Data descriptor %s is '
                            'written to, but only given to nested SDFG as an '
                            'input connector' % node.data, sdfg, state_id, nid)

        if (isinstance(node, nd.ConsumeEntry)
                and "IN_stream" not in node.in_connectors):
            raise InvalidSDFGNodeError(
                "Consume entry node must have an input stream", sdfg, state_id,
                nid)
        if (isinstance(node, nd.ConsumeEntry)
                and "OUT_stream" not in node.out_connectors):
            raise InvalidSDFGNodeError(
                "Consume entry node must have an internal stream",
                sdfg,
                state_id,
                nid,
            )   

        # Connector tests
        ########################################
        # Check for duplicate connector names (unless it's a nested SDFG)
        if (len(node.in_connectors.keys() & node.out_connectors.keys()) > 0
                and not isinstance(node, (nd.NestedSDFG, nd.LibraryNode))):
            dups = node.in_connectors.keys() & node.out_connectors.keys()
            raise InvalidSDFGNodeError("Duplicate connectors: " + str(dups),
                                       sdfg, state_id, nid)

        # Check for connectors that are also array/symbol names
        if isinstance(node, nd.Tasklet):
            for conn in node.in_connectors.keys():
                if conn in sdfg.arrays or conn in symbols:
                    raise InvalidSDFGNodeError(
                        f"Input connector {conn} already "
                        "defined as array or symbol", sdfg, state_id, nid)
            for conn in node.out_connectors.keys():
                if conn in sdfg.arrays or conn in symbols:
                    raise InvalidSDFGNodeError(
                        f"Output connector {conn} already "
                        "defined as array or symbol", sdfg, state_id, nid)

        # Check for dangling connectors (incoming)
        for conn in node.in_connectors:
            incoming_edges = 0
            for e in state.in_edges(node):
                # Connector found
                if e.dst_conn == conn:
                    incoming_edges += 1

            if incoming_edges == 0:
                raise InvalidSDFGNodeError("Dangling in-connector %s" % conn,
                                           sdfg, state_id, nid)
            # Connectors may have only one incoming edge
            # Due to input connectors of scope exit, this is only correct
            # in some cases:
            if incoming_edges > 1 and not isinstance(node, nd.ExitNode):
                raise InvalidSDFGNodeError(
                    "Connector '%s' cannot have more "
                    "than one incoming edge, found %d" % (conn, incoming_edges),
                    sdfg,
                    state_id,
                    nid,
                )

        # Check for dangling connectors (outgoing)
        for conn in node.out_connectors:
            outgoing_edges = 0
            for e in state.out_edges(node):
                # Connector found
                if e.src_conn == conn:
                    outgoing_edges += 1

            if outgoing_edges == 0:
                raise InvalidSDFGNodeError("Dangling out-connector %s" % conn,
                                           sdfg, state_id, nid)

            # In case of scope exit or code node, only one outgoing edge per
            # connector is allowed.
            if outgoing_edges > 1 and isinstance(node,
                                                 (nd.ExitNode, nd.CodeNode)):
                raise InvalidSDFGNodeError(
                    "Connector '%s' cannot have more "
                    "than one outgoing edge, found %d" % (conn, outgoing_edges),
                    sdfg,
                    state_id,
                    nid,
                )

        # Check for edges to nonexistent connectors
        for e in state.in_edges(node):
            if e.dst_conn is not None and e.dst_conn not in node.in_connectors:
                raise InvalidSDFGNodeError(
                    ("Memlet %s leading to " + "nonexistent connector %s") %
                    (str(e.data), e.dst_conn),
                    sdfg,
                    state_id,
                    nid,
                )
        for e in state.out_edges(node):
            if e.src_conn is not None and e.src_conn not in node.out_connectors:
                raise InvalidSDFGNodeError(
                    ("Memlet %s coming from " + "nonexistent connector %s") %
                    (str(e.data), e.src_conn),
                    sdfg,
                    state_id,
                    nid,
                )
        ########################################

    # Memlet checks
    scope = state.scope_dict()
    for eid, e in enumerate(state.edges()):
        # Edge validation
        try:
            e.data.validate(sdfg, state)
        except InvalidSDFGError:
            raise
        except Exception as ex:
            raise InvalidSDFGEdgeError("Edge validation failed: " + str(ex),
                                       sdfg, state_id, eid)

        # For every memlet, obtain its full path in the DFG
        path = state.memlet_path(e)
        src_node = path[0].src
        dst_node = path[-1].dst

        # Check if memlet data matches src or dst nodes
        if (e.data.data is not None and (isinstance(src_node, nd.AccessNode)
                                         or isinstance(dst_node, nd.AccessNode))
                and (not isinstance(src_node, nd.AccessNode)
                     or e.data.data != src_node.data)
                and (not isinstance(dst_node, nd.AccessNode)
                     or e.data.data != dst_node.data)):
            raise InvalidSDFGEdgeError(
                "Memlet data does not match source or destination "
                "data nodes)",
                sdfg,
                state_id,
                eid,
            )

        # Check memlet subset validity with respect to source/destination nodes
        if e.data.data is not None and e.data.allow_oob == False:
            subset_node = (dst_node if isinstance(dst_node, nd.AccessNode)
                           and e.data.data == dst_node.data else src_node)
            other_subset_node = (dst_node if isinstance(dst_node, nd.AccessNode)
                                 and e.data.data != dst_node.data else src_node)

            def getArrayCheckOffset(checknode : nd.AccessNode, arr : data.Array):
                if(checknode.data in local_hbmmultibank_arrays):
                    currentarrayoffset = [0]
                    currentarrayoffset.extend(arr.shape)
                else:
                    currentarrayoffset = arr.offset
                return currentarrayoffset

            def getArrayCheckShape(checknode : nd.AccessNode, arr : data.Array):
                if(checknode.data in local_hbmmultibank_arrays):
                    mbinfo = local_hbmmultibank_arrays[checknode.data]
                    currentarrayshape = [mbinfo['numbank']]
                    currentarrayshape.extend(hbm_multibank_expansion.getHBMTrueShape(
                        arr.shape, mbinfo['splitaxes'], mbinfo['splitcount']
                    ))
                else:
                    currentarrayshape = arr.shape
                return tuple(currentarrayshape)
                
            def findMapDefiningSymbol(symbolstr : str, startnode : nd.Node):
                scopeHere = scope[startnode]
                if(scopeHere is None):
                    return None
                if(symbolstr in scopeHere.params):
                    return scopeHere
                return findMapDefiningSymbol(symbolstr, scopeHere)

            def checkIsValidMultiBankSubset(checksubset, associatedEdge, arrayinfo):
                low, high, stride = checksubset[0]
                allowed_number_types = low.is_Atom and high.is_Atom and stride.is_Atom
                isSymbolicIndex = low.is_Symbol and high.is_Symbol and low==high and str(stride) == "1"
                #allowed_number_types = allowed_number_types and ((low.is_Symbol or high.is_Symbol) and (not isSymbolicIndex))
                allowed_number_types = allowed_number_types and not ((low.is_Symbol or high.is_Symbol) and (not isSymbolicIndex))
                allowed_number_types = allowed_number_types and not stride.is_Symbol
                if(not allowed_number_types):
                    raise InvalidSDFGEdgeError("a HBM index may only be either a constant range" 
                            "of integers, or a variable bound by a map that is unrolled", sdfg, state_id, eid)
                if(isSymbolicIndex):
                    deepScopeNode = associatedEdge.src
                    if(scope[associatedEdge.dst] == associatedEdge.src):
                        deepScopeNode = associatedEdge.dst
                    map = findMapDefiningSymbol(str(low), deepScopeNode)
                    if(map is None):
                        raise InvalidSDFGEdgeError("symbol is used as HBM index, but a map" 
                            f"defining symbol {str(low)} was not found", sdfg, state_id, eid)
                    generated_values = hbm_multibank_expansion.get_unroll_map_properties(state, map)
                    if(isinstance(generated_values, str)):
                        raise InvalidSDFGEdgeError("Cannot unroll the map defining the" 
                            f"hbm index variable {str(low)}, because: {generated_values}", 
                            sdfg, state_id, eid)
                    lowerbound = generated_values[0]
                    upperbound = generated_values[-1]
                else:
                    lowerbound = int(str(low))
                    upperbound = int(str(high))
                if not (lowerbound >= 0 and upperbound < arrayinfo['numbank']):
                    raise InvalidSDFGEdgeError("HBM index is out of bounds", sdfg, state_id, eid)
                if(isinstance(associatedEdge.dst, nd.Tasklet) or isinstance(associatedEdge.src, nd.Tasklet)):
                    if(lowerbound != upperbound and not isSymbolicIndex):
                        raise InvalidSDFGEdgeError("A memlet accessing a range of HBM banks may not"
                                "be attached to a Tasklet", sdfg, state_id, eid)

            if isinstance(subset_node, nd.AccessNode):
                arr = sdfg.arrays[subset_node.data]
                # Dimensionality
                expectedDim = len(arr.shape)
                if(subset_node.data in local_hbmmultibank_arrays):
                    expectedDim += 1
                if (e.data.subset.dims() != expectedDim):
                    raise InvalidSDFGEdgeError(
                        "Memlet subset does not match node dimension "
                        "(expected %d, got %d)" %
                        (expectedDim, e.data.subset.dims()),
                        sdfg,
                        state_id,
                        eid,
                    )
                if(subset_node.data in local_hbmmultibank_arrays):
                    checkIsValidMultiBankSubset(e.data.subset, e, 
                                local_hbmmultibank_arrays[subset_node.data])
                    

                # Bounds
                arroffset = getArrayCheckOffset(subset_node, arr)
                if any(((minel + off) < 0) == True for minel, off in zip(
                        e.data.subset.min_element(), arroffset)):
                    raise InvalidSDFGEdgeError(
                        "Memlet subset negative out-of-bounds", sdfg, state_id,
                        eid)
                if any(((maxel + off) >= s) == True for maxel, s, off in zip(
                        e.data.subset.max_element(), getArrayCheckShape(subset_node, arr), arroffset)):
                    raise InvalidSDFGEdgeError("Memlet subset out-of-bounds",
                                               sdfg, state_id, eid)

            # Test other_subset as well
            if e.data.other_subset is not None and isinstance(
                    other_subset_node, nd.AccessNode):
                arr = sdfg.arrays[other_subset_node.data]
                # Dimensionality
                expectedOtherDim = len(arr.shape)
                if other_subset_node.data in local_hbmmultibank_arrays:
                    expectedOtherDim += 1
                if e.data.other_subset.dims() != expectedOtherDim:
                    raise InvalidSDFGEdgeError(
                        "Memlet other_subset does not match node dimension "
                        "(expected %d, got %d)" %
                        (expectedOtherDim, e.data.other_subset.dims()),
                        sdfg,
                        state_id,
                        eid,
                    )
                if(other_subset_node.data in local_hbmmultibank_arrays):
                    checkIsValidMultiBankSubset(e.data.other_subset, e, 
                        local_hbmmultibank_arrays[other_subset_node.data])

                # Bounds
                arroffset = getArrayCheckOffset(other_subset_node, arr)
                if any(((minel + off) < 0) == True for minel, off in zip(
                        e.data.other_subset.min_element(), arroffset)):
                    raise InvalidSDFGEdgeError(
                        "Memlet other_subset negative out-of-bounds",
                        sdfg,
                        state_id,
                        eid,
                    )
                if any(((maxel + off) >= s) == True for maxel, s, off in zip(
                        e.data.other_subset.max_element(), getArrayCheckShape(other_subset_node, arr),
                        arroffset)):
                    raise InvalidSDFGEdgeError(
                        "Memlet other_subset out-of-bounds", sdfg, state_id,
                        eid)

            # Test subset and other_subset for undefined symbols
            if Config.get_bool('experimental', 'validate_undefs') or True:
                # TODO: Traverse by scopes and accumulate data
                defined_symbols = state.symbols_defined_at(e.dst)
                undefs = (e.data.subset.free_symbols -
                          set(defined_symbols.keys()))
                if len(undefs) > 0:
                    raise InvalidSDFGEdgeError(
                        'Undefined symbols %s found in memlet subset' % undefs,
                        sdfg, state_id, eid)
                if e.data.other_subset is not None:
                    undefs = (e.data.other_subset.free_symbols -
                              set(defined_symbols.keys()))
                    if len(undefs) > 0:
                        raise InvalidSDFGEdgeError(
                            'Undefined symbols %s found in memlet '
                            'other_subset' % undefs, sdfg, state_id, eid)
        #######################################

        # Memlet path scope lifetime checks
        # If scope(src) == scope(dst): OK
        if scope[src_node] == scope[dst_node] or src_node == scope[dst_node]:
            pass
        # If scope(src) contains scope(dst), then src must be a data node,
        # unless the memlet is empty in order to connect to a scope
        elif scope_contains_scope(scope, src_node, dst_node):
            pass
        # If scope(dst) contains scope(src), then dst must be a data node,
        # unless the memlet is empty in order to connect to a scope
        elif scope_contains_scope(scope, dst_node, src_node):
            if not isinstance(dst_node, nd.AccessNode):
                if e.data.is_empty() and isinstance(dst_node, nd.ExitNode):
                    pass
                else:
                    raise InvalidSDFGEdgeError(
                        f"Memlet creates an invalid path (sink node {dst_node}"
                        " should be a data node)", sdfg, state_id, eid)
        # If scope(dst) is disjoint from scope(src), it's an illegal memlet
        else:
            raise InvalidSDFGEdgeError("Illegal memlet between disjoint scopes",
                                       sdfg, state_id, eid)

        #TODO: Isn't this already checked at 485?
        """
        # Check dimensionality of memory access
        if isinstance(e.data.subset, (sbs.Range, sbs.Indices)):
            if e.data.subset.dims() != len(sdfg.arrays[e.data.data].shape):
                raise InvalidSDFGEdgeError(
                    "Memlet subset uses the wrong dimensions"
                    " (%dD for a %dD data node)" %
                    (e.data.subset.dims(), len(sdfg.arrays[e.data.data].shape)),
                    sdfg,
                    state_id,
                    eid,
                )
        """

        # Verify that source and destination subsets contain the same
        # number of elements
        if not e.data.allow_oob and e.data.other_subset is not None and not (
            (isinstance(src_node, nd.AccessNode)
             and isinstance(sdfg.arrays[src_node.data], dt.Stream)) or
            (isinstance(dst_node, nd.AccessNode)
             and isinstance(sdfg.arrays[dst_node.data], dt.Stream))):
            src_expr = (e.data.src_subset.num_elements() *
                        sdfg.arrays[src_node.data].veclen)
            dst_expr = (e.data.dst_subset.num_elements() *
                        sdfg.arrays[dst_node.data].veclen)
            if src_node.data in local_hbmmultibank_arrays or dst_node.data in local_hbmmultibank_arrays:
                pass #TODO: This does not yet work, because we have to add the assumption that N / numbanks == floor(N / numbanks)
            elif symbolic.inequal_symbols(src_expr, dst_expr):
                raise InvalidSDFGEdgeError(
                    'Dimensionality mismatch between src/dst subsets', sdfg,
                    state_id, eid)
    ########################################


###########################################
# Exception classes


class InvalidSDFGError(Exception):
    """ A class of exceptions thrown when SDFG validation fails. """
    def __init__(self, message: str, sdfg, state_id):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id

    def to_json(self):
        return dict(message=self.message,
                    sdfg_id=self.sdfg.sdfg_id,
                    state_id=self.state_id)

    def __str__(self):
        if self.state_id is not None:
            state = self.sdfg.nodes()[self.state_id]
            return "%s (at state %s)" % (self.message, str(state.label))
        else:
            return "%s" % self.message


class InvalidSDFGInterstateEdgeError(InvalidSDFGError):
    """ Exceptions of invalid inter-state edges in an SDFG. """
    def __init__(self, message: str, sdfg, edge_id):
        self.message = message
        self.sdfg = sdfg
        self.edge_id = edge_id

    def to_json(self):
        return dict(message=self.message,
                    sdfg_id=self.sdfg.sdfg_id,
                    isedge_id=self.edge_id)

    def __str__(self):
        if self.edge_id is not None:
            e = self.sdfg.edges()[self.edge_id]
            edgestr = ' (at edge "%s" (%s -> %s)' % (
                e.data.label,
                str(e.src),
                str(e.dst),
            )
        else:
            edgestr = ""

        return "%s%s" % (self.message, edgestr)


class InvalidSDFGNodeError(InvalidSDFGError):
    """ Exceptions of invalid nodes in an SDFG state. """
    def __init__(self, message: str, sdfg, state_id, node_id):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.node_id = node_id

    def to_json(self):
        return dict(message=self.message,
                    sdfg_id=self.sdfg.sdfg_id,
                    state_id=self.state_id,
                    node_id=self.node_id)

    def __str__(self):
        state = self.sdfg.nodes()[self.state_id]

        if self.node_id is not None:
            node = state.nodes()[self.node_id]
            nodestr = ", node %s" % str(node)
        else:
            nodestr = ""

        return "%s (at state %s%s)" % (self.message, str(state.label), nodestr)


class NodeNotExpandedError(InvalidSDFGNodeError):
    """
    Exception that is raised whenever a library node was not expanded
    before code generation.
    """
    def __init__(self, sdfg: 'dace.sdfg.SDFG', state_id: int, node_id: int):
        super().__init__('Library node not expanded', sdfg, state_id, node_id)


class InvalidSDFGEdgeError(InvalidSDFGError):
    """ Exceptions of invalid edges in an SDFG state. """
    def __init__(self, message: str, sdfg, state_id, edge_id):
        self.message = message
        self.sdfg = sdfg
        self.state_id = state_id
        self.edge_id = edge_id

    def to_json(self):
        return dict(message=self.message,
                    sdfg_id=self.sdfg.sdfg_id,
                    state_id=self.state_id,
                    edge_id=self.edge_id)

    def __str__(self):
        state = self.sdfg.nodes()[self.state_id]

        if self.edge_id is not None:
            e = state.edges()[self.edge_id]
            edgestr = ", edge %s (%s:%s -> %s:%s)" % (
                str(e.data),
                str(e.src),
                e.src_conn,
                str(e.dst),
                e.dst_conn,
            )
        else:
            edgestr = ""

        return "%s (at state %s%s)" % (self.message, str(state.label), edgestr)
