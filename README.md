# Flame.jl

Toy code for compression corner with lux NN model for real gas and chemical reaction.

- `CUDA.jl` for parallelization
- 2D, with high order scheme
- `Lux.jl` trained model combined with `Cantera`

## Roadmap
- [x] Proper viscous
- [x] Species transport
- [x] Integrate Cantera and Lux
- [ ] Shuffle and stream optimization