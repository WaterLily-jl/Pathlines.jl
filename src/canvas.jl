import Makie

"""
    PathlineCanvas

Pixel buffer and rendering parameters for canvas-based pathline visualization.

Use `fade!(pc)` to lerp all pixels toward the background color each frame,
and `draw!(pc, pos, pos⁰, dt)` to rasterize particle step segments into the buffer.
The `canvas` field is a `Matrix{RGBAf}` suitable for use as a `Makie.Observable`.
"""
struct PathlineCanvas
    canvas      ::Matrix{Makie.RGBAf}   # px × py pixel buffer
    bgcolor     ::Makie.RGBAf
    fade        ::Float32
    cmap_colors                          # 256-entry colormap LUT
    clo         ::Float32                # speed-range low
    chi         ::Float32                # speed-range high
    sx          ::Float32                # sim → canvas x scale
    sy          ::Float32                # sim → canvas y scale
    figsize     ::Tuple{Int,Int}
end

"""
    PathlineCanvas(nx, ny; bgcolor=:black, fadetau=0.2,
                   colormap=:plasma, colorrange=(0, 3),
                   figsize=nothing, resolution=nothing)

Construct a `PathlineCanvas` for a simulation grid of interior size `(nx, ny)`.

- `bgcolor`    : canvas background color (faded toward each frame).
- `fadetau`    : trail persistence time constant, in convective time units (L/U). Creates pathlines O(`fadetau*L`) long.
- `colormap`   : Makie colormap for speed-based segment coloring.
- `colorrange` : `(lo, hi)` speed values mapped across the colormap.
- `figsize`    : figure pixel size `(width, height)`; defaults to `(1200, 1200*ny/nx)`.
- `resolution` : canvas pixel size `(px, py)`; defaults to `figsize`.
"""
function PathlineCanvas(nx::Int, ny::Int;
                        bgcolor=:black, fadetau=0.2,
                        colormap=:plasma, colorrange=(0, 3),
                        figsize=nothing, resolution=nothing)
    fs     = isnothing(figsize)    ? (1200, (1200*ny)÷nx) : figsize
    px, py = isnothing(resolution) ? fs          : resolution
    bg     = _rgba(bgcolor)
    canvas = fill(bg, px, py)
    cmap   = Makie.resample_cmap(colormap, 256)
    PathlineCanvas(canvas, bg, Float32(fadetau), cmap,
                   Float32(colorrange[1]), Float32(colorrange[2]),
                   Float32(px / nx), Float32(py / ny), fs)
end

# ── public API ────────────────────────────────────────────────────────────────

"""
    fade!(pc::PathlineCanvas)

Lerp every canvas pixel toward the background color by the canvas fade factor.
"""
function fade!(pc::PathlineCanvas, dt=1f0)
    α, bg = 1 - exp(-Float32(dt/pc.fade)), pc.bgcolor
    @inbounds @simd for k in eachindex(pc.canvas)
        c = pc.canvas[k]
        pc.canvas[k] = Makie.RGBAf(c.r + (bg.r - c.r)*α,
                                    c.g + (bg.g - c.g)*α,
                                    c.b + (bg.b - c.b)*α, 1f0)
    end
end

"""
    draw!(pc::PathlineCanvas, pos, pos⁰, dt)

Rasterize particle step segments `pos⁰ → pos` into the canvas buffer,
colouring by speed via the canvas colormap. `dt` is the integration timestep
used only for speed normalization.
"""
function draw!(pc::PathlineCanvas, pos, pos⁰, dt::Float32)
    _draw_segments!(pc.canvas, pos, pos⁰, pc.cmap_colors,
                    pc.clo, pc.chi, pc.sx, pc.sy, dt)
end

# ── private rendering primitives ─────────────────────────────────────────────

function _rgba(c)
    c4 = Makie.to_color(c)
    Makie.RGBAf(Float32(Makie.red(c4)), Float32(Makie.green(c4)),
                Float32(Makie.blue(c4)), 1f0)
end

@inline function _splat!(canvas, fi::Float32, fj::Float32,
                          color::Makie.RGBAf, px::Int, py::Int)
    i0 = floor(Int, fi);  i1 = i0 + 1
    j0 = floor(Int, fj);  j1 = j0 + 1
    wi = fi - Float32(i0);  wj = fj - Float32(j0)
    _blend_pixel!(canvas, i0, j0, color, (1f0-wi)*(1f0-wj), px, py)
    _blend_pixel!(canvas, i1, j0, color,        wi*(1f0-wj), px, py)
    _blend_pixel!(canvas, i0, j1, color, (1f0-wi)*wj,        px, py)
    _blend_pixel!(canvas, i1, j1, color,        wi*wj,        px, py)
end

@inline function _blend_pixel!(canvas, i::Int, j::Int,
                                color::Makie.RGBAf, w::Float32, px::Int, py::Int)
    (i < 1 || i > px || j < 1 || j > py) && return
    @inbounds c = canvas[i, j]
    @inbounds canvas[i, j] = Makie.RGBAf(c.r + (color.r - c.r)*w,
                                          c.g + (color.g - c.g)*w,
                                          c.b + (color.b - c.b)*w, 1f0)
end

function _draw_segments!(canvas::Matrix{Makie.RGBAf}, pos, pos⁰,
                          cmap_colors, lo::Float32, hi::Float32,
                          sx::Float32, sy::Float32, dt::Float32)
    px, py = size(canvas)
    ncmap  = length(cmap_colors)
    @inbounds @simd for k in eachindex(pos)
        p0, p1 = pos⁰[k], pos[k]
        dx, dy = p1[1]-p0[1], p1[2]-p0[2]
        speed  = sqrt(dx*dx + dy*dy) / dt
        t      = clamp((speed - lo) / (hi - lo + 1f-6), 0f0, 1f0)
        cidx   = clamp(round(Int, t*(ncmap-1)) + 1, 1, ncmap)
        color  = Makie.RGBAf(cmap_colors[cidx])
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
