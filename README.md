# Flame.jl

Toy code for compression corner with lux NN model for real gas and chemical reaction.

- `CUDA.jl` for parallelization
- 2D, low order scheme
- `Lux.jl` trained model combined with `Cantera`

## Roadmap
- [x] Proper viscous
- [ ] Shuffle and stream optimization
- [ ] Species transport
- [ ] Integrate Cantera and Lux
- [ ] 3D and MPI