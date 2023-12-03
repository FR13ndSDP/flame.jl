# Flame.jl

Toy code for compression corner with lux NN model for real gas and chemical reaction.

- `CUDA.jl` for parallelization
- 2D (for now), with high order scheme
- `Lux.jl` trained model combined with `Cantera`

## Roadmap
- [x] Proper viscous
- [ ] Species transport
- [ ] Integrate Cantera and Lux
- [ ] 3D and MPI
- [ ] Shuffle and stream optimization