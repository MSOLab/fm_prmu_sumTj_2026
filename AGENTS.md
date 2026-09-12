# Repository Guidance

Permutation flow shop scheduling to minimize total tardiness. `README.md` has
the problem statement and the solver line-up.

## Working conventions

- Run Python through `uv run` and `uv run python`, never bare `python` or
  `python3`. The dependencies live in the uv-managed environment, so a bare
  interpreter may be the wrong one or miss installed packages. This is enforced,
  not advisory: the `PreToolUse` hook `.claude/hooks/uv-only-python.sh` denies
  bare invocations.
- Investigation scripts live in `checks/`; run them the same way.
- Deferred work lives in the root `TODO.md`, one `##` entry per item that states
  the problem and a proposed approach. There is no second TODO file.
- Plans live in `plans/`, named `YYYYMMDD_descriptive_snake_case.md`.
- Experiment and investigation write-ups live in `docs/report/`. A write-up with
  several files goes in a `YYYYMMDD/` directory, a single document in
  `YYYYMMDD_descriptive_snake_case.md`.

## Baseline Agent Rules

Apply communication rules to reader-facing text, including responses and
documentation, and coding rules to code changes. More specific instructions
override these rules; report material trade-offs when useful.

### Communication

#### Plain Language

Write for the intended reader's knowledge and purpose. Apply these principles
together to responses, documentation, instructions, and other reader-facing text:

- Relevant: Include what the reader needs to know or do; remove unnecessary detail.
- Findable: Order information around the reader's task; use headings, lists, or
  tables when they help the reader locate information.
- Understandable: Use familiar words, direct sentences, and consistent terms;
  explain technical terms when needed for the reader's understanding.
- Usable: Make actions, conditions, and expected results clear when relevant;
  check that the reader can use the information without guessing.

Preserve facts, conditions, exceptions, and uncertainty when simplifying.
Keep necessary technical precision; brevity alone is not the goal.
Use reader feedback when available; claim reader testing only if it occurred.

### Coding Principles

#### KISS: Keep It Simple, Stupid

Prefer the simplest implementation that meets current requirements;
add complexity only when demanded.

#### YAGNI: You ain't gonna need it

Do not add features, abstractions, or configurability until needed.

#### DRY: Don't repeat yourself

Extract duplication only when it represents the same knowledge,
not merely similar-looking code.

#### SOLID principles

In OO code, keep responsibilities focused, interfaces small, substitutions
valid, and dependencies aimed at stable abstractions; do not abstract for
speculative extension.

### Architectural Patterns

#### Single source of truth architecture

Create and update each data element in one authoritative location;
derived copies read from it.

### Best Practices

#### TDD: Test-Driven Development

When changing behavior, write a test and confirm it fails for the expected
reason, implement the minimum, then improve with tests green. Do not require
tests for docs or trivial config changes.

#### BDD: Behavior-Driven Development

When useful for requirements, describe stakeholder-visible behavior as one
Given-When-Then scenario using shared domain terms.

#### Contract-First Development

Before implementing a public API or cross-component boundary, define its
machine-readable contract, errors, and invariants; treat incompatible changes
as breaking.
