# DynamicDiscreteSamplers

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://LilithHafner.github.io/DynamicDiscreteSamplers.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LilithHafner.github.io/DynamicDiscreteSamplers.jl/dev/)
[![Build Status](https://github.com/LilithHafner/DynamicDiscreteSamplers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LilithHafner/DynamicDiscreteSamplers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/LilithHafner/DynamicDiscreteSamplers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LilithHafner/DynamicDiscreteSamplers.jl) <!--
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/D/DynamicDiscreteSamplers.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/D/DynamicDiscreteSamplers.html) -->
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`DynamicDiscreteSamplers.jl` implements efficient samplers which can be used to sample from a dynamic discrete distribution, supporting removal, addition and sampling of elements in constant time. 

The key features of this package are

- Exact Sampling: The probability of sampling an index is exactly proportional to its weight;
- Fast Sampling: O(1) worst-case expected runtime for drawing a sample;
- Fast Updates: O(1) worst-case amortized runtime for updating any weight;
- Memory Efficient: O(n) space complexity, where n is the number of weights;
- Dynamic Sizing: Supports samplers that can be resized;
- Practical Performance: low constant factors, making it fast in practice.

The package exports two main types which conform to the `AbstractArray` API:

- `FixedSizeWeights`: For a static collection of weights which can be updated but not resized;
- `ResizableWeights`: For a collection of weights which can grow or shrink.

```julia
julia> using Random, DynamicDiscreteSamplers

julia> rng = Xoshiro(42);

julia> w = ResizableWeights([10.0, 50.0, 5.0, 35.0])
4-element ResizableWeights:
 10.0
 50.0
  5.0
 35.0

julia> println(rand(rng, w, 10))
[2, 4, 4, 2, 2, 2, 4, 2, 4, 2]

julia> w[1] = 100.0;

julia> println(rand(rng, w, 10))
[2, 1, 2, 1, 1, 2, 4, 1, 2, 2]

julia> resize!(w, 6);

julia> w[6] = 200.0;

julia> println(rand(rng, w, 10))
[6, 2, 6, 6, 6, 1, 2, 6, 6, 1]
```
