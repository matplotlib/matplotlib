# This file is part of colorspacious
# Copyright (C) 2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import numpy as np

from collections import namedtuple, defaultdict

__all__ = ["Edge", "MATCH", "ANY", "TransformGraph"]

################################################################
# Basic types and sentinels
################################################################

# We give these a string id for ease of debugging, but they are really
# singleton sentinels compared via is/is not.
class Placeholder(object):
    def __init__(self, id):
        self.id = id

    def __repr__(self):  # pragma: no cover
        return "<{0}>".format(self.id)

MATCH = Placeholder("MATCH")
ANY = Placeholder("ANY")

START = Placeholder("START")
END = Placeholder("END")

Edge = namedtuple("Edge", ["start", "end", "transform"])
Path = namedtuple("Path", ["nodes", "transforms"])

################################################################
# Invariant checking
################################################################

def assert_(b):
    if not b:
        raise AssertionError

def check_node(node, allowed_placeholders):
    assert_(isinstance(node, dict))
    assert_(isinstance(node.get("name"), str))
    for v in node.values():
        if isinstance(v, Placeholder):
            assert_(v in allowed_placeholders)

def test_check_node():
    from nose.tools import assert_raises
    assert_raises(AssertionError, check_node, {}, [])
    check_node({"name": "asdf"}, [])
    assert_raises(AssertionError, check_node, {"name": 1}, [])
    check_node({"name": "asdf", "a": "b"}, [])
    assert_raises(AssertionError, check_node,
                  {"name": "asdf", "a": MATCH}, [ANY])
    check_node({"name": "asdf", "a": ANY, "b": "c"}, [ANY])

# edge_node = edge node, can have MATCH or ANY or concrete values
# used in primitive edges
def check_edge_node(edge_node):
    check_node(edge_node, set([MATCH, ANY]))

# path_node = path node, can have START or END or concrete values
# used in precomputed paths
def check_path_node(path_node):
    check_node(path_node, set([START, END]))

# concrete_node = concrete node, can only have concrete values
# used in queries
def check_concrete_node(concrete_node):
    check_node(concrete_node, set())

def check_edge(edge):
    check_edge_node(edge.start)
    check_edge_node(edge.end)
    NOTHING = object()
    for k in set(edge.start).union(edge.end):
        start_v = edge.start.get(k, NOTHING)
        end_v = edge.end.get(k, NOTHING)
        # Every _MATCH must have a matching _MATCH.
        if MATCH in (start_v, end_v):
            assert_(start_v is end_v is MATCH)
        # For simplicity, ANY must not match anything.
        # In principle you could have a transform
        #    {"foo": "concrete_value"} -> {"foo": ANY}
        # or even
        #    {"foo": ANY} -> {"foo": ANY}
        # where the transform can turn any setting of foo into any other. But
        # we currently don't need this, it complicates things, and makes it
        # harder to pass the values to the transform, so YAGNI.
        if ANY in (start_v, end_v):
            assert_(set([start_v, end_v]) == set([ANY, NOTHING]))

def test_check_edge():
    from nose.tools import assert_raises

    check_edge(Edge({"name": "foo"}, {"name": "bar"}, "fake transform"))
    check_edge(Edge({"name": "foo", "start_any": ANY, "match": MATCH},
                    {"name": "bar", "end_any": ANY, "match": MATCH},
                    "fake transform"))
    for bad in [
            Edge({"name": 1}, {"name": "bar"}, "fake transform"),
            Edge({"name": "foo"}, {"name": 1}, "fake transform"),
            Edge({"name": "foo"}, {"name": "bar", "other": START},
                 "fake transform"),
            Edge({"name": "foo", "a": ANY}, {"name": "bar", "a": ANY},
                 "fake transform"),
            Edge({"name": "foo", "a": MATCH}, {"name": "bar", "a": "asdf"},
                 "fake transform"),
            Edge({"name": "foo", "a": "asdf"}, {"name": "bar", "a": MATCH},
                 "fake transform"),
            ]:
        assert_raises(AssertionError, check_edge, bad)

def check_path(path):
    for node in path.nodes:
        check_path_node(node)
    for start, end in zip(path.nodes[:-1], path.nodes[1:]):
        for k in set(start).union(end):
            if k != "name" and k in start and k in end:
                assert_(start[k] == end[k])
            # START and END must have an unbroken chain from the beginning or
            # end of the path respectively, b/c the only way they can legally
            # appear in the middle is via MATCH constraints.
            if start.get(k) is END:
                assert_(end.get(k) is END)
            if end.get(k) is START:
                assert_(start.get(k) is START)
    assert_(len(path.nodes) == len(path.transforms) + 1)

def test_check_path():
    from nose.tools import assert_raises

    check_path(Path(({"name": "foo"}, {"name": "bar"}), ("t1",)))
    check_path(Path(({"name": "foo"},
                     {"name": "bar"},
                     {"name": "baz"}),
                    ("t1", "t2")))
    for bad in [
            # must be path nodes
            Path([{"name": 1}, {"name": "bar"}], ["t1"]),
            Path([{"name": "foo"}, {"name": 1}], ["t1"]),
            Path([{"name": "foo", "a": ANY}, {"name": "bar"}], ["t1"]),
            Path([{"name": "foo"}, {"name": "bar", "b": MATCH}], ["t1"]),
            # no changing values on a transition
            Path([{"name": "foo", "a": 1}, {"name": "bar", "a": 2}], ["t1"]),
            # no dropping END
            Path([{"name": "foo", "a": END}, {"name": "bar"}], ["t1"]),
            Path([{"name": "foo", "a": END}, {"name": "bar", "a": 2}], ["t1"]),
            # no spontaneous generation of START
            Path([{"name": "foo"}, {"name": "bar", "a": START}], ["t1"]),
            Path([{"name": "foo", "a": 1}, {"name": "bar", "a": START}], ["t1"]),
            # length mismatches
            Path([{"name": "foo"}, {"name": "bar"}], []),
            Path([{"name": "foo"}, {"name": "bar"}], ["t1", "t2"]),
            ]:
        assert_raises(AssertionError, check_path, bad)

################################################################
# Dict manipulation utilities
################################################################

# Utilities for replacing values in a dict in-place
def _replace_values(d, replacements):
    # replacements is {old_value: new_value}
    for k, v in d.items():
        if d[k] in replacements:
            d[k] = replacements[v]

def test__replace_values():
    d = {"a":  1, "b": 2, "c":  1}
    _replace_values(d, {1: 11})
    assert_(d == {"a": 11, "b": 2, "c": 11})

def _fill_values_from(d, marker, template):
    # replaces all instances of 'marker' in d with corresponding value from
    # 'template'
    for k, v in d.items():
        if d[k] is marker:
            d[k] = template[k]

def test__fill_values_from():
    d = {"a": 1, "b": 2, "c": 1}
    _fill_values_from(d, 1, {"a": "a1", "b": "b1", "c": "c1"})
    assert_(d == {"a": "a1", "b": 2, "c": "c1"})

################################################################
# Graph calculations
################################################################

def trivial_path(edge_node):
    path_node = dict(edge_node)
    _replace_values(path_node, {MATCH: START, ANY: START})
    return Path([path_node], [])

def test_trivial_path():
    assert_(trivial_path({"name": "start",
                          "foo": MATCH,
                          "bar": ANY,
                          "baz": 1})
            == Path([{"name": "start",
                      "foo": START,
                      "bar": START,
                      "baz": 1}],
                    []))

def try_extend_path(path, edge):
    # Returns a new Path, or None if this edge can't extend this path.

    # First, check that the last node of the path is compatible with the first
    # node of the edge. This might require imposing new constraints on the
    # path, e.g. if our path is
    #
    #   {"name": "a", "x": START} -> {"name": "b", "x": START}
    #
    # and we want to extend it with an edge
    #
    #   {"name": "b", "x": 0} -> {"name": "c"}
    #
    # then that should produce the path
    #
    #   {"name": "a", "x": 0} -> {"name": "b", "x": 0} -> {"name": "c"}
    #
    # where we match the overlapping nodes and then propagated the new
    # constraint back.
    if path.nodes[-1].keys() != edge.start.keys():
        return None
    path_fill_ins = defaultdict(dict)
    for k in path.nodes[-1]:
        path_value = path.nodes[-1][k]
        edge_value = edge.start[k]
        if path_value is START:
            # START matches anything, but if the edge has a concrete value
            # then we need to add that constraint
            if not isinstance(edge_value, Placeholder):
                path_fill_ins[START][k] = edge_value
        elif path_value is END:
            # END matches concrete values and MATCH, but not ANY (because if
            # it matched ANY, then the END chain would be lost)
            if isinstance(edge_value, Placeholder):
                if edge_value is ANY:
                    return None
                else:
                    assert_(edge_value is MATCH)
            else:
                path_fill_ins[END][k] = edge_value
        else:
            assert_(not isinstance(path_value, Placeholder))
            # Concrete values match placeholders, and identical concrete
            # values
            if (not isinstance(edge_value, Placeholder)
                and path_value != edge_value):
                return None
    # propagate any new constraints backwards through the path
    if path_fill_ins:
        new_nodes = []
        for node in path.nodes:
            new_node = dict(node)
            for placeholder, replacements in path_fill_ins.items():
                for key, value in replacements.items():
                    if new_node.get(key) is placeholder:
                        new_node[key] = value
            new_nodes.append(new_node)
        path = Path(new_nodes, path.transforms)
    # now extend the path forward by adding a new node
    new_path_end = dict(edge.end)
    # Propagate forward MATCH values
    _fill_values_from(new_path_end, MATCH, path.nodes[-1])
    # Add any new END values
    _replace_values(new_path_end, {ANY: END})
    return Path(path.nodes + [new_path_end],
                path.transforms + [edge.transform])

def test_try_extend_path():
    assert_(try_extend_path(
        Path([{"name": "s", "a": 1}], []),
        Edge({"name": "s", "a": 1}, {"name": "e", "b": 2}, "t1"))
            == Path([{"name": "s", "a": 1}, {"name": "e", "b": 2}],
                    ["t1"]))

    # "a" mismatch
    assert_(try_extend_path(
        Path([{"name": "s", "a": 1}], []),
        Edge({"name": "s", "a": 2}, {"name": "e", "b": 2}, "t1"))
            is None)

    # keys mismatch
    assert_(try_extend_path(
        Path([{"name": "s", "a": 1}], []),
        Edge({"name": "s", "b": 2}, {"name": "e", "b": 2}, "t1"))
            is None)

    # match ANY
    assert_(try_extend_path(
        Path([{"name": "s", "a": 1}], []),
        Edge({"name": "s", "a": ANY}, {"name": "e", "b": 2}, "t1"))
            == Path([{"name": "s", "a": 1}, {"name": "e", "b": 2}],
                    ["t1"]))

    # ANY -> END
    assert_(try_extend_path(
        Path([{"name": "s", "a": 1}], []),
        Edge({"name": "s", "a": 1}, {"name": "e", "b": ANY}, "t1"))
            == Path([{"name": "s", "a": 1}, {"name": "e", "b": END}],
                    ["t1"]))

    # propagate value through MATCH
    assert_(try_extend_path(
        Path([{"name": "s", "a": 1}], []),
        Edge({"name": "s", "a": MATCH}, {"name": "e", "a": MATCH}, "t1"))
            == Path([{"name": "s", "a": 1}, {"name": "e", "a": 1}],
                    ["t1"]))

    # propagate placeholder through MATCH
    assert_(try_extend_path(
        Path([{"name": "s", "a": START}], []),
        Edge({"name": "s", "a": MATCH}, {"name": "e", "a": MATCH}, "t1"))
            == Path([{"name": "s", "a": START}, {"name": "e", "a": START}],
                    ["t1"]))

    assert_(try_extend_path(
        Path([{"name": "s", "a": END}], []),
        Edge({"name": "s", "a": MATCH}, {"name": "e", "a": MATCH}, "t1"))
            == Path([{"name": "s", "a": END}, {"name": "e", "a": END}],
                    ["t1"]))

    # losing an END is no good
    assert_(try_extend_path(
        Path([{"name": "s", "a": END}], []),
        Edge({"name": "s", "a": ANY}, {"name": "e", "b": 1}, "t1"))
            is None)

    # replacing an END or START with a concrete value works
    assert_(try_extend_path(
        Path([{"name": "s", "a": START, "b": START}], []),
        Edge({"name": "s", "a": 1, "b": ANY}, {"name": "e", "c": 1}, "t1"))
            == Path([{"name": "s", "a": 1, "b": START},
                     {"name": "e", "c": 1}],
                    ["t1"]))

    assert_(try_extend_path(
        Path([{"name": "s", "a": END, "b": END}], []),
        Edge({"name": "s", "a": 1, "b": 2}, {"name": "e", "c": 1}, "t1"))
            == Path([{"name": "s", "a": 1, "b": 2},
                     {"name": "e", "c": 1}],
                    ["t1"]))

    # and this reaches backwards through the path
    assert_(try_extend_path(
        Path([{"name": "1", "a": START, "b": START},
              {"name": "2", "a": START},
              {"name": "3", "a": START, "b": END}],
             ["t1", "t2"]),
        Edge({"name": "3", "a": 1, "b": 2}, {"name": "4", "c": 1}, "t3"))
            == Path([{"name": "1", "a": 1, "b": START},
                     {"name": "2", "a": 1},
                     {"name": "3", "a": 1, "b": 2},
                     {"name": "4", "c": 1}],
                    ["t1", "t2", "t3"]))


def pairwise_shortest_paths(edges):
    # start node name -> [Edge]
    edges_from = defaultdict(list)

    # (frozen start node, frozen end node) -> Path
    shortest_paths = {}
    # list of Path objects
    todo_next = []

    # Simple implementation of breadth-first-search:
    def observe_path(path):
        check_path(path)
        #print("  new path: %r" % (path,))
        key = (tuple(path.nodes[0].items()), tuple(path.nodes[-1].items()))
        if key not in shortest_paths:
            #print("    ...accepted")
            shortest_paths[key] = path
            todo_next.append(path)
        else:
            #print("    ...rejected")
            pass

    for edge in edges:
        edges_from[edge.start["name"]].append(edge)
        observe_path(trivial_path(edge.start))

    while todo_next:
        todo_now = todo_next
        todo_next = []
        for path in todo_now:
            #print("extending: %s" % (path,))
            for edge in edges_from[path.nodes[-1]["name"]]:
                new_path = try_extend_path(path, edge)
                #print("  + %s -> %s" % (edge, new_path))
                if new_path is not None:
                    observe_path(new_path)


    shortest_paths_by_name = {}
    for path in shortest_paths.values():
        key = (path.nodes[0]["name"], path.nodes[-1]["name"])
        shortest_paths_by_name.setdefault(key, []).append(path)

    return shortest_paths_by_name

def test_pairwise_shortest_paths():
    # a -> b, b -> c
    assert_(pairwise_shortest_paths([
        Edge({"name": "a"}, {"name": "b"}, "ab"),
        Edge({"name": "b"}, {"name": "c"}, "bc"),
    ]) == {
        ("a", "a"): [Path([{"name": "a"}], [])],
        ("b", "b"): [Path([{"name": "b"}], [])],
        ("a", "b"): [Path([{"name": "a"}, {"name": "b"}], ["ab"])],
        ("b", "c"): [Path([{"name": "b"}, {"name": "c"}], ["bc"])],
        ("a", "c"): [Path([{"name": "a"}, {"name": "b"}, {"name": "c"}],
                          ["ab", "bc"])],
        })

    # a -> b, b -> c, a -> c
    # finds shortest a -> c path
    assert_(pairwise_shortest_paths([
        Edge({"name": "a"}, {"name": "b"}, "ab"),
        Edge({"name": "b"}, {"name": "c"}, "bc"),
        Edge({"name": "a"}, {"name": "c"}, "ac"),
    ]) == {
        ("a", "a"): [Path([{"name": "a"}], [])],
        ("b", "b"): [Path([{"name": "b"}], [])],
        ("a", "b"): [Path([{"name": "a"}, {"name": "b"}], ["ab"])],
        ("b", "c"): [Path([{"name": "b"}, {"name": "c"}], ["bc"])],
        ("a", "c"): [Path([{"name": "a"}, {"name": "c"}], ["ac"])],
        })

    # MATCH propagation
    assert_(pairwise_shortest_paths([
        Edge({"name": "a", "x": MATCH}, {"name": "b", "x": MATCH}, "ab"),
        Edge({"name": "b", "x": ANY}, {"name": "c"}, "bc"),
    ]) == {
        ("a", "a"): [Path([{"name": "a", "x": START}], [])],
        ("b", "b"): [Path([{"name": "b", "x": START}], [])],
        ("a", "b"): [Path([{"name": "a", "x": START},
                           {"name": "b", "x": START}],
                          ["ab"])],
        ("b", "c"): [Path([{"name": "b", "x": START},
                           {"name": "c"}],
                          ["bc"])],
        ("a", "c"): [Path([{"name": "a", "x": START},
                           {"name": "b", "x": START},
                           {"name": "c"}],
                          ["ab", "bc"])],
        })

    # a -> b, b -> c, a -> c
    # but a->c direct edge requires specific setting for x, so we get two
    # different paths
    paths = pairwise_shortest_paths([
        Edge({"name": "a", "x": ANY}, {"name": "b"}, "ab"),
        Edge({"name": "b"}, {"name": "c", "x": ANY}, "bc"),
        Edge({"name": "a", "x": 1}, {"name": "c", "x": 1}, "ac"),
    ])
    assert_(len(paths[("a", "c")]) == 3)
    assert_(Path([{"name": "a", "x": START},
                  {"name": "b"},
                  {"name": "c", "x": END}],
                 ["ab", "bc"])
            in paths[("a", "c")])
    assert_(Path([{"name": "a", "x": 1},
                  {"name": "c", "x": 1}],
                 ["ac"])
            in paths[("a", "c")])
    # The third one is a redundant a|x=1 -> b -> c|x=END


################################################################
# Paths + concrete nodes
################################################################

def concretize_path_node(path_node, start_concrete_node, end_concrete_node):
    concrete = dict(path_node)
    _fill_values_from(concrete, START, start_concrete_node)
    _fill_values_from(concrete, END, end_concrete_node)
    return concrete

def test_concretize_path_node():
    assert_(concretize_path_node(
        {"name": "foo",   "a": START, "b": END,  "c": "given"},
        {"name": "start", "a": "sa",  "b": "sb", "c": "sc"},
        {"name": "end",   "a": "ea",  "b": "eb", "c": "ec"})
            == {"name": "foo", "a": "sa", "b": "eb", "c": "given"})

def safe_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b

def transform_kwargs(start_concrete_node, end_concrete_node):
    kwargs = {}
    for k, v in start_concrete_node.items():
        if k != "name":
            kwargs[k] = v
    for k, v in end_concrete_node.items():
        if k != "name":
            if k in kwargs:
                assert_(safe_equal(kwargs[k], v))
            else:
                kwargs[k] = v
    return kwargs

def test_transform_kwargs():
    from nose.tools import assert_raises

    assert_(transform_kwargs({"name": "start"}, {"name": "end"}) == {})
    assert_(transform_kwargs(
        {"name": "start", "a": 1, "b": 2},
        {"name": "end", "b": 2, "c": 3}
        ) == {"a": 1, "b": 2, "c": 3})
    assert_raises(AssertionError, transform_kwargs,
                  {"name": "start", "a": 1},
                  {"name": "end",   "a": 2})

def path_matches(path, desired_concrete_start, desired_concrete_end):
    # check that path[0] and path[-1] match concrete_start and concrete_end
    # and also that this holds even after filling in the values
    path_concrete_start = concretize_path_node(path.nodes[0],
                                               desired_concrete_start,
                                               desired_concrete_end)
    path_concrete_end = concretize_path_node(path.nodes[-1],
                                             desired_concrete_start,
                                             desired_concrete_end)
    return ((path_concrete_start == desired_concrete_start)
            and (path_concrete_end == desired_concrete_end))

def test_path_matches():
    assert_(path_matches(Path(({"name": "start"}, {"name": "end"}), ("t1",)),
                         {"name": "start"}, {"name": "end"}))
    assert_(path_matches(
        Path(({"name": "start", "a": START},
              {"name": "end", "b": END}),
             ("t1",)),
        {"name": "start", "a": 1},
        {"name": "end", "b": 2}))

    assert_(path_matches(
        Path(({"name": "start", "a": START},
              {"name": "end", "a": START}),
             ("t1",)),
        {"name": "start", "a": 1},
        {"name": "end", "a": 1}))

    assert_(not path_matches(
        Path(({"name": "start", "a": START},
              {"name": "end", "b": END}),
             ("t1",)),
        {"name": "start_mismatch", "a": 1},
        {"name": "end", "b": 2}))

    assert_(not path_matches(
        Path(({"name": "start", "a": 1},
              {"name": "end", "b": 2}),
             ("t1",)),
        {"name": "start", "a": 1},
        {"name": "end", "b": 22}))

    assert_(not path_matches(
        Path(({"name": "start", "a": START},
              {"name": "end", "a": START}),
             ("t1",)),
        {"name": "start", "a": 1},
        {"name": "end", "a": 2}))

class Transform(object):
    def __init__(self, nodes, transforms, kwargses):
        self.nodes = nodes
        self._transforms = transforms
        self._kwargses = kwargses

    def __call__(self, x):
        for transform, kwargs in zip(self._transforms, self._kwargses):
            x = transform(x, **kwargs)
        return x

def test_Transform():
    log = []
    def t1(x, **kwargs):
        log.append(("t1", x, kwargs))
        return x * 2
    def t2(x, **kwargs):
        log.append(("t2", x, kwargs))
        return x * 2

    t = Transform(["a", "b", "c"],
                  [t1, t2],
                  [{"t1_arg": 1}, {"t2_arg": 2}])
    assert t("x") == "xxxx"
    assert log == [
        ("t1", "x", {"t1_arg": 1}),
        ("t2", "xx", {"t2_arg": 2}),
        ]

################################################################
# Top-level object
################################################################

class TransformGraph(object):
    def __init__(self, edges, rank_constraints=[]):
        # This is what's used for actual calculations
        self._shortest_paths = pairwise_shortest_paths(edges)

        # All the rest is used only for the dot output
        self._rank_constraints = rank_constraints
        self._edges = edges
        self._nodes = []
        seen_names = set()
        for edge in edges:
            check_edge(edge)
            for node in (edge.start, edge.end):
                if node["name"] not in seen_names:
                    seen_names.add(node["name"])
                    self._nodes.append(node)

    def get_transform(self, start, end):
        check_concrete_node(start)
        check_concrete_node(end)
        if start == end:
            return Transform([start], [], [])

        start_name = start["name"]
        end_name = end["name"]
        best_path = None
        for path in self._shortest_paths.get((start_name, end_name), []):
            if path_matches(path, start, end):
                if best_path is None or len(best_path.nodes) > len(path.nodes):
                    best_path = path
        if best_path is None:
            raise ValueError("No path found from %r -> %r" % (start, end))
        concrete_nodes = []
        for path_node in best_path.nodes:
            concrete_nodes.append(concretize_path_node(path_node, start, end))
        kwargses = []
        for i in range(len(concrete_nodes) - 1):
            kwargs = transform_kwargs(concrete_nodes[i],
                                      concrete_nodes[i + 1])
            kwargses.append(kwargs)
        return Transform(concrete_nodes, best_path.transforms, kwargses)

    def dump_dot(self, f): # pragma: no cover
        f.write("digraph {\n")
        for node in self._nodes:
            # Laziness: assumes names don't need more quoting
            attr_names = set(node)
            attr_names.remove("name")
            html = "<b>%s</b>" % (node["name"],)
            for attr_name in sorted(attr_names):
                html += "<br/>  <i>%s</i>" % (attr_name,)
            f.write("  \"%s\" [ label=<%s> ]\n" % (node["name"], html))
        for edge in self._edges:
            # FIXME: label edges with attribute information
            # taillabel=<...>, headlabel=<...>, ...
            f.write("  \"%s\" -> \"%s\"\n"
                    % (edge.start["name"], edge.end["name"]))
        for rank_constraint in self._rank_constraints:
            f.write("  {rank=same; "
                    + ", ".join(["\"%s\"" % (c,) for c in rank_constraint])
                    + "}\n")
        f.write("}\n")
