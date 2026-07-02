module Pathlines

include("Particles.jl")
export Particles, update!

include("canvas.jl")
export PathlineCanvas, fade!, draw!

end