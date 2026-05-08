import GLMakie

# ── color helpers ─────────────────────────────────────────────────────────────

"""Convert any Makie-parseable colour to a fully-opaque `RGBAf`."""
function _rgba(c)
    c4 = GLMakie.to_color(c)
    GLMakie.RGBAf(Float32(GLMakie.red(c4)), Float32(GLMakie.green(c4)),
                  Float32(GLMakie.blue(c4)), 1f0)
end

# ── canvas operations ─────────────────────────────────────────────────────────

"""
Lerp every canvas pixel toward `bg` by factor `α` (0 = no fade, 1 = instant wipe).
"""
function _fade_canvas!(canvas::Matrix{GLMakie.RGBAf}, bg::GLMakie.RGBAf, α::Float32)
    @inbounds for k in eachindex(canvas)
        c = canvas[k]
        canvas[k] = GLMakie.RGBAf(c.r + (bg.r - c.r)*α,
                                   c.g + (bg.g - c.g)*α,
                                   c.b + (bg.b - c.b)*α, 1f0)
    end
end

"""
    _splat!(canvas, fi, fj, color, px, py)

Bilinear splat: distribute `color` into the four pixels surrounding the
fractional canvas coordinate `(fi, fj)`, weighted by proximity.
Each neighbouring pixel is lerp'd toward `color` with its area weight, so
a pixel hit exactly at its centre gets full coverage.  This gives
sub-pixel anti-aliased line segments.
"""
@inline function _splat!(canvas, fi::Float32, fj::Float32,
                          color::GLMakie.RGBAf, px::Int, py::Int)
    i0 = floor(Int, fi);  i1 = i0 + 1
    j0 = floor(Int, fj);  j1 = j0 + 1
    wi = fi - Float32(i0);  wj = fj - Float32(j0)
    _blend_pixel!(canvas, i0, j0, color, (1f0-wi)*(1f0-wj), px, py)
    _blend_pixel!(canvas, i1, j0, color,        wi*(1f0-wj), px, py)
    _blend_pixel!(canvas, i0, j1, color, (1f0-wi)*wj,        px, py)
    _blend_pixel!(canvas, i1, j1, color,        wi*wj,        px, py)
end

@inline function _blend_pixel!(canvas, i::Int, j::Int,
                                color::GLMakie.RGBAf, w::Float32, px::Int, py::Int)
    (i < 1 || i > px || j < 1 || j > py) && return
    @inbounds c = canvas[i, j]
    @inbounds canvas[i, j] = GLMakie.RGBAf(c.r + (color.r - c.r)*w,
                                            c.g + (color.g - c.g)*w,
                                            c.b + (color.b - c.b)*w, 1f0)
end

"""
    _draw_segments!(canvas, pos, pos⁰, cmap_colors, lo, hi, sx, sy)

Rasterize the particle step segments `pos⁰ → pos` into `canvas` using
bilinear splatting (sub-pixel anti-aliasing).  Segment colour is mapped
from speed via `cmap_colors`.  `sx` / `sy` convert sim grid coordinates
to canvas pixel coordinates.
"""
function _draw_segments!(canvas::Matrix{GLMakie.RGBAf}, pos, pos⁰,
                          cmap_colors, lo::Float32, hi::Float32,
                          sx::Float32, sy::Float32, dt::Float32)
    px, py = size(canvas)
    ncmap  = length(cmap_colors)
    @inbounds for k in eachindex(pos)
        p0, p1 = pos⁰[k], pos[k]
        dx, dy = p1[1]-p0[1], p1[2]-p0[2]
        speed  = sqrt(dx*dx + dy*dy)/dt

        # Colour from speed
        t    = clamp((speed - lo) / (hi - lo + 1f-6), 0f0, 1f0)
        cidx = clamp(round(Int, t*(ncmap-1)) + 1, 1, ncmap)
        color = GLMakie.RGBAf(cmap_colors[cidx])

        # Step count: 2 sub-pixel steps per canvas pixel traversed so
        # bilinear splats overlap slightly and produce continuous coverage.
        pix_len = sqrt((dx*sx)^2 + (dy*sy)^2)
        nsteps  = max(1, ceil(Int, 2f0 * pix_len))
        inv_ns  = 1f0 / nsteps

        for s in 0:nsteps
            f  = s * inv_ns
            fi = clamp((p0[1] + f*dx) * sx, 1f0, Float32(px))
            fj = clamp((p0[2] + f*dy) * sy, 1f0, Float32(py))
            _splat!(canvas, fi, fj, color, px, py)
        end
    end
end

# ── PathViz ───────────────────────────────────────────────────────────────────

"""
    PathViz(sim; N, life, mem, body, bgcolor, figsize, resolution,
                 fadealpha, colormap, colorrange, bodycolor)

Canvas-based pathline visualisation for a WaterLily simulation.

Uses a single `Matrix{RGBAf}` pixel buffer as the background: exactly one
`image!` plot and (optionally) one body-contour layer on top, regardless of
how many frames are rendered.  Scene size is O(1).

Each frame the canvas is lerp-faded toward `bgcolor`, then new particle
segments are anti-aliased into it via bilinear splatting.  The body contour
is a separate Makie layer, unaffected by fading.

# Keyword arguments
- `N=10_000`          : number of Lagrangian particles
- `life=UInt(255)`    : particle lifetime in `update!` steps
- `mem=Array`         : array constructor (`CUDA.CuArray` for GPU)
- `body=true`         : overlay the body SDF contour
- `bgcolor=:black`    : background colour (canvas fades toward this)
- `figsize`           : `(width, height)` in pixels; default `(4nx, 4ny)`
- `resolution`        : canvas pixel size `(px, py)`; defaults to `figsize`
- `fadealpha=0.2`     : per-frame fade strength (0 = no fade, 1 = instant wipe)
- `colormap=:plasma`  : Makie colormap for velocity-magnitude colouring
- `colorrange=(0, 3)` : `(lo, hi)` speed range mapped across the colormap
- `bodycolor=:white`  : colour of the body SDF contour overlay

Typical usage inside `GLMakie.record`:
```julia
GLMakie.record(viz.fig, "out.mp4", 0.1:0.1:30) do t
    Pathlines.step!(viz, sim, t)
end
```
"""
struct PathViz
    p           ::Particles
    fig         ::GLMakie.Figure
    ax          ::GLMakie.Axis
    _canvas     ::Matrix{GLMakie.RGBAf}
    _ocanvas    ::GLMakie.Observable
    _bgcolor    ::GLMakie.RGBAf
    _fade       ::Float32
    _cmap_colors              # Vector of RGBAf colormap LUT
    _clo        ::Float32
    _chi        ::Float32
    _bodycolor                # nothing when body=false
    _σbody                    # CPU body SDF buffer (nothing when body=false)
    _oσbody                   # Observable wrapping _σbody (nothing when body=false)
    _sx         ::Float32     # sim→canvas x scale factor
    _sy         ::Float32     # sim→canvas y scale factor
    _sim_nx     ::Int
    _sim_ny     ::Int
end

function PathViz(sim; N=10_000, life=UInt(255), mem=Array,
                 body=true, bgcolor=:black, figsize=nothing,
                 fadealpha=0.2, resolution=nothing,
                 colormap=:plasma, colorrange=(0, 3),
                 bodycolor=:white)
    σ      = sim.flow.σ
    nx, ny = size(σ, 1), size(σ, 2)
    fs     = isnothing(figsize) ? (4nx, 4ny) : figsize
    px, py = isnothing(resolution) ? fs : resolution

    p       = Particles(N, σ; mem, life)
    bg_rgba = _rgba(bgcolor)
    canvas  = fill(bg_rgba, px, py)
    ocanvas = GLMakie.Observable(canvas)
    sx, sy  = Float32(px / nx), Float32(py / ny)

    # 256-entry colormap lookup table
    cmap_colors = GLMakie.resample_cmap(colormap, 256)

    fig = GLMakie.Figure(size=fs)
    ax  = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, limits=(1, nx, 1, ny))
    GLMakie.hidedecorations!(ax)

    # Single image plot — canvas pixel [i,j] covers sim coord (i, j)
    GLMakie.image!(ax, (1, nx), (1, ny), ocanvas)

    # Body contour on top — separate Makie layer, unaffected by canvas fading
    σb, oσb = if body
        buf = Array(σ[WaterLily.inside(σ)])
        obs = GLMakie.Observable(buf)
        WaterLily.plot_body_obs!(ax, obs; color=bodycolor)
        buf, obs
    else
        nothing, nothing
    end

    PathViz(p, fig, ax, canvas, ocanvas, bg_rgba, Float32(fadealpha),
            cmap_colors, Float32(colorrange[1]), Float32(colorrange[2]),
            body ? bodycolor : nothing, σb, oσb, sx, sy, nx, ny)
end

# ── update / step API ─────────────────────────────────────────────────────────

"""
    update!(v::PathViz, sim)

Advance particles one sim step and splat new segments into the canvas.
Does **not** notify the Observable; call [`step!`](@ref) to also flush to display.
"""
function update!(v::PathViz, sim)
    update!(v.p, sim)
    pos  = Array(v.p.position)
    pos⁰ = Array(v.p.position⁰)
    dt = sim.flow.Δt[end-1]
    _draw_segments!(v._canvas, pos, pos⁰, v._cmap_colors, v._clo, v._chi, v._sx, v._sy, dt)
end

"""
    step!(v::PathViz, sim, t; remeasure=false)

Advance the simulation and particles up to time `t`.  Fades the canvas,
rasterizes all new particle segments with anti-aliasing, then notifies the
display.  Updates the body SDF observable when `remeasure=true`.
"""
function step!(v::PathViz, sim, t; remeasure=false)
    _fade_canvas!(v._canvas, v._bgcolor, v._fade)
    while sim_time(sim) < t
        sim_step!(sim; remeasure)
        update!(v, sim)
    end
    GLMakie.notify(v._ocanvas)
    if !isnothing(v._oσbody) && remeasure
        WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
        copyto!(v._σbody, sim.flow.σ[WaterLily.inside(sim.flow.σ)])
        GLMakie.notify(v._oσbody)
    end
end
