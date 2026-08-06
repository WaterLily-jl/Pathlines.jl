# SailDemo App

Interactive sail simulation built with LilyPad + Pathlines. Arrow keys control angle of attack (up/down) and sheet tension (left/right).

## Building the executable

Run these commands from the root of the `Pathlines.jl` repository on the target machine. Requires Julia installed; no GPU needed.

```julia
# 1. Resolve dependencies (~5 min first time)
julia --project=app -e 'using Pkg; Pkg.instantiate()'

# 2. Compile the app (~10–20 min)
julia --project=app -e '
  using PackageCompiler
  create_app("app", "app/SailApp"; incremental=false, include_lazy_artifacts=true)
'
```

## Running

```
app\SailApp\bin\SailDemo
```

Or double-click `SailDemo.exe` in `app/SailApp/bin/`. No Julia installation required on the end machine — the `SailApp/` directory is self-contained.

## Notes

- `[sources]` in `Project.toml` resolves `Pathlines` from the parent directory (`..`). The build must be run from inside the `Pathlines.jl/` repo root.
- The compiled `SailApp/` directory can be copied anywhere after building.
- First launch may be slightly slower than subsequent ones due to remaining JIT paths.
