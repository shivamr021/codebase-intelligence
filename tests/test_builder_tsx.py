from core.graph.builder import build_dependency_graph

APP = ('import { Item } from "./Item";\n\n'
       'export function App({ items }: { items: string[] }) {\n'
       '  return <ul>{items.map((i) => <Item key={i} />)}</ul>;\n}\n')
ITEM = 'export function Item() {\n  return <li />;\n}\n'


def test_tsx_import_of_sibling_component_gives_edge(tmp_path):
    # GUARD (green before the fix): TypeScript grammar keeps this top import too. Break: import extraction or resolution for .tsx dropped.
    files = []
    for name, text in (("App.tsx", APP), ("Item.tsx", ITEM)):
        (tmp_path / name).write_text(text)
        files.append({"path": str(tmp_path / name), "rel_path": f"src/{name}", "language": "typescript", "size": 1})
    assert list(build_dependency_graph(files).edges) == [("src/App.tsx", "src/Item.tsx")]


LOGO = 'export function Logo() {\n  return <img src={require("./Icon")} alt="logo" />;\n}\n'
ICON = 'export const Icon = "icon";\n'


def test_tsx_require_inside_jsx_attribute_gives_edge(tmp_path):
    # Break: builder parses .tsx with the plain TypeScript grammar; it misreads <img ...> and loses the require, so no edge.
    files = []
    for name, text in (("Logo.tsx", LOGO), ("Icon.tsx", ICON)):
        (tmp_path / name).write_text(text)
        files.append({"path": str(tmp_path / name), "rel_path": f"src/{name}", "language": "typescript", "size": 1})
    assert list(build_dependency_graph(files).edges) == [("src/Logo.tsx", "src/Icon.tsx")]
