module Pathlines

include("Particles.jl")
export Particles, update!

include("canvas.jl")
export PathlineCanvas, fade!, draw!

# draw pathlines with `WaterLily.viz!` whenever Pathlines is loaded
include("viz.jl")
__init__() = WaterLily._pathlines_viz_hook[] = _pathlines_setup

end