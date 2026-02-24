# CLAUDE.md

## Project

dpvt-experiments: Deep neural networks for Phylogenetics Via Traversals — experiments.

## Commands

- Use `pixi run` to execute commands (e.g., `pixi run pytest`, `pixi run black`)
- Run tests: `pixi run pytest`
- Format code: `pixi run black`

## Domain conventions

### ete3 Tree edge counting

In this codebase, edge labels are stored as flat lists corresponding to a
preorder traversal of an ete3 Tree. The first 2 positions (root and first child)
are masked/unused. Therefore, `total_edges = len(tree) - 2` is intentional:
`len(tree)` returns the number of leaves, and subtracting 2 accounts for the
masked root and first-child positions. This is a domain convention, not a bug.
